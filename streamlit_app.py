"""
Smart Queue Monitor — Main Streamlit Application
Supports:
1. 📹 Live Webcam (WebRTC)
2. 🎬 Demo / Uploaded Video (YOLOv8 + DeepSORT on recorded video)
3. 📸 Snapshot Camera Input
"""
import os
import sys
import tempfile
import threading
import time
from collections import deque

# Prevent permission issues with Ultralytics settings in restricted environments
if "YOLO_CONFIG_DIR" not in os.environ:
    os.environ["YOLO_CONFIG_DIR"] = tempfile.gettempdir()

# Ensure pkg_resources is available for deep_sort_realtime
try:
    import pkg_resources
except ImportError:
    try:
        import setuptools
        import pkg_resources
    except ImportError:
        pass

import av
import cv2
import numpy as np
import streamlit as st
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer
from ultralytics import YOLO


# =============================================================================
# CONFIG
# =============================================================================
class Config:
    def __init__(self):
        self.model_path = "yolov8n.pt"
        self.use_gpu = True
        self.yolo_conf = 0.45
        self.min_person_height = 60
        self.frame_skip = 3
        self.process_width = 320
        self.queue_roi = [50, 200, 400, 500]
        self.service_roi = [420, 200, 650, 450]
        self.deep_sort_max_age = 30
        self.deep_sort_n_init = 2


# =============================================================================
# QUEUE MONITOR ENGINE
# =============================================================================
class SmartQueueMonitor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = "cuda" if cfg.use_gpu and torch.cuda.is_available() else "cpu"

        self.model = YOLO(cfg.model_path)
        if self.device == "cuda":
            self.model = self.model.to(self.device).half()
        else:
            self.model = self.model.to(self.device)

        self.tracker = DeepSort(
            max_age=cfg.deep_sort_max_age,
            n_init=cfg.deep_sort_n_init,
            embedder_gpu=self.device == "cuda",
            half=self.device == "cuda",
        )

        self.entry_times: dict[int, float] = {}
        self.total_served = 0
        self.avg_service_time: float | None = None
        self.alpha = 0.3
        self.frame_count = 0
        self._lock = threading.Lock()

        # Cached state for overlays
        self.cached_boxes: list[tuple[int, int, int, int, int]] = []
        self.cached_stats = {
            "queue": 0, "service": 0, "served": 0, "wait": 0.0,
            "status": "LOW", "proc_ms": 0.0,
        }
        self.box_history: dict[int, deque[tuple[int, int, int, int]]] = {}
        self.proc_ms_ema = 0.0

    @staticmethod
    def fix_roi(roi: list[int]) -> list[int]:
        roi[0] = max(0, roi[0])
        roi[1] = max(0, roi[1])
        roi[2] = max(roi[0] + 10, roi[2])
        roi[3] = max(roi[1] + 10, roi[3])
        return roi

    def _run_detection(self, display: np.ndarray) -> None:
        cfg = self.cfg
        t0 = time.time()

        pw = cfg.process_width
        yolo_frame = cv2.resize(display, (pw, int(display.shape[0] * pw / display.shape[1])))
        results = self.model(yolo_frame, imgsz=pw, conf=cfg.yolo_conf, verbose=False)

        scale = display.shape[1] / pw
        detections = []
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) != 0:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1, x2, y2 = (int(x1 * scale), int(y1 * scale),
                                  int(x2 * scale), int(y2 * scale))
                if (y2 - y1) < cfg.min_person_height:
                    continue
                detections.append(([x1, y1, x2 - x1, y2 - y1], float(box.conf[0]), "person"))

        tracks = self.tracker.update_tracks(detections, frame=display)

        queue_count = 0
        service_count = 0
        now = time.time()
        q, s = cfg.queue_roi, cfg.service_roi
        active_ids = set()
        new_cached = []

        for t in tracks:
            if not t.is_confirmed() or t.time_since_update > 1:
                continue

            track_id = t.track_id
            active_ids.add(track_id)

            x1, y1, x2, y2 = map(int, t.to_ltrb())

            hist = self.box_history.setdefault(track_id, deque(maxlen=3))
            hist.append((x1, y1, x2, y2))
            x1 = int(sum(b[0] for b in hist) / len(hist))
            y1 = int(sum(b[1] for b in hist) / len(hist))
            x2 = int(sum(b[2] for b in hist) / len(hist))
            y2 = int(sum(b[3] for b in hist) / len(hist))

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if q[0] < cx < q[2] and q[1] < cy < q[3]:
                queue_count += 1
                if track_id not in self.entry_times:
                    self.entry_times[track_id] = now
            elif s[0] < cx < s[2] and s[1] < cy < s[3]:
                service_count += 1
                if track_id in self.entry_times:
                    individual_wait = now - self.entry_times[track_id]
                    if 2 < individual_wait < 120:
                        if self.avg_service_time is None:
                            self.avg_service_time = individual_wait
                        else:
                            self.avg_service_time = (
                                self.alpha * individual_wait
                                + (1 - self.alpha) * self.avg_service_time
                            )
                        self.total_served += 1
                        del self.entry_times[track_id]

            new_cached.append((track_id, x1, y1, x2, y2))

        for tid in list(self.box_history.keys()):
            if tid not in active_ids:
                del self.box_history[tid]

        predicted_wait = 0.0
        if self.avg_service_time and queue_count > 0:
            predicted_wait = queue_count * self.avg_service_time / max(service_count, 1)

        if queue_count > 8:
            status = "HIGH"
        elif queue_count > 4:
            status = "MEDIUM"
        else:
            status = "LOW"

        self.cached_boxes = new_cached
        proc_ms = (time.time() - t0) * 1000
        self.proc_ms_ema = proc_ms if self.proc_ms_ema == 0 else 0.7 * proc_ms + 0.3 * self.proc_ms_ema
        self.cached_stats = {
            "queue": queue_count,
            "service": service_count,
            "served": self.total_served,
            "wait": predicted_wait,
            "status": status,
            "proc_ms": self.proc_ms_ema,
        }

    def _draw_overlays(self, display: np.ndarray) -> None:
        cfg = self.cfg

        # Zones (BGR)
        cv2.rectangle(display, tuple(cfg.queue_roi[:2]), tuple(cfg.queue_roi[2:]), (0x00, 0x4D, 0xFF), 2)
        cv2.rectangle(display, tuple(cfg.service_roi[:2]), tuple(cfg.service_roi[2:]), (0xC8, 0xC8, 0x00), 2)

        # Tracked boxes (BGR)
        for track_id, x1, y1, x2, y2 in self.cached_boxes:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            q, s = cfg.queue_roi, cfg.service_roi
            if q[0] < cx < q[2] and q[1] < cy < q[3]:
                color = (0x00, 0x7A, 0xFF)
            elif s[0] < cx < s[2] and s[1] < cy < s[3]:
                color = (0xE8, 0xE8, 0x00)
            else:
                color = (0x4A, 0xD0, 0x4A)

            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            label = f"ID {track_id}"
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(display, (x1, y1 - 25), (x1 + tw + 8, y1), color, -1)
            cv2.putText(display, label, (x1 + 4, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.circle(display, (cx, cy), 4, color, -1)

        self._draw_info_panel(display)

        if self.proc_ms_ema > 0:
            cv2.putText(display, f"inference {self.proc_ms_ema:.0f} ms",
                        (700, 530),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0xAA, 0xAA, 0xAA), 1, cv2.LINE_AA)

    def _draw_info_panel(self, frame: np.ndarray) -> None:
        s = self.cached_stats
        x, y = 10, 10
        box_w, box_h = 250, 155

        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (0x0A, 0x0C, 0x0F), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (0x1E, 0x25, 0x30), 1)

        def put(label: str, value: str, yy: int, color: tuple[int, int, int]) -> None:
            cv2.putText(frame, f"{label}  {value}", (x + 12, yy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        yy = y + 26
        step = 22

        status_colors = {
            "LOW": (0x4A, 0xD0, 0x4A),
            "MEDIUM": (0x4B, 0xA8, 0xE8),
            "HIGH": (0x00, 0x4D, 0xFF)
        }
        sc = status_colors.get(s["status"], (0xAA, 0xAA, 0xAA))

        put("Queue", str(s["queue"]), yy, (0xC8, 0xC8, 0x00)); yy += step
        put("Wait", f"{int(s['wait'])} s", yy, (0x4B, 0xA8, 0xE8)); yy += step
        put("Status", s["status"], yy, sc); yy += step
        put("In Service", str(s["service"]), yy, (0xE8, 0xE8, 0x00)); yy += step
        put("Served", str(s["served"]), yy, (0x4A, 0xD0, 0x4A)); yy += step

        if self.avg_service_time:
            put("Avg Service", f"{self.avg_service_time:.1f} s", yy, (0xB0, 0x9B, 0x8A))

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        with self._lock:
            self.frame_count += 1
            display = cv2.resize(frame, (960, 540))

            if self.frame_count % self.cfg.frame_skip == 0:
                self._run_detection(display)

            self._draw_overlays(display)
            return display

    def get_stats(self) -> dict:
        with self._lock:
            return self.cached_stats.copy()


class QueueVideoProcessor:
    def __init__(self, monitor: SmartQueueMonitor):
        self.monitor = monitor

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        processed = self.monitor.process_frame(img)
        return av.VideoFrame.from_ndarray(processed, format="bgr24")

    def get_stats(self) -> dict:
        return self.monitor.get_stats()


# =============================================================================
# STREAMLIT UI
# =============================================================================
st.set_page_config(
    page_title="Smart Queue Monitor",
    page_icon="⏱️",
    layout="wide",
)

def show_badge(label: str, icon: str = None, color: str = None):
    if hasattr(st, "badge"):
        st.badge(label, icon=icon, color=color)
    else:
        st.markdown(f"**[{label}]**")

def get_rtc_config() -> RTCConfiguration:
    ice_servers = [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
    ]

    turn_url = None
    turn_username = None
    turn_password = None

    try:
        if hasattr(st, "secrets"):
            turn_url = st.secrets.get("TURN_URL")
            turn_username = st.secrets.get("TURN_USERNAME")
            turn_password = st.secrets.get("TURN_PASSWORD")
    except Exception:
        pass

    turn_url = turn_url or os.getenv("TURN_URL")
    turn_username = turn_username or os.getenv("TURN_USERNAME")
    turn_password = turn_password or os.getenv("TURN_PASSWORD")

    if turn_url and turn_username and turn_password:
        ice_servers.append({
            "urls": [turn_url],
            "username": turn_username,
            "credential": turn_password,
        })

    return RTCConfiguration({"iceServers": ice_servers})


st.title("Smart Queue Monitor")
st.caption("Live AI monitoring powered by YOLOv8 and DeepSORT tracking")

# Mode Selection
app_mode = st.radio(
    "Select Input Mode:",
    ["📹 Live Webcam (WebRTC)", "🎬 Demo / Uploaded Video", "📸 Camera Snapshot"],
    horizontal=True,
)

# ---- Sidebar Configuration ----
with st.sidebar:
    st.header("⚙️ Configuration")

    has_gpu = torch.cuda.is_available()
    show_badge(
        "CUDA GPU" if has_gpu else "CPU Only",
        icon=":material/check_circle:" if has_gpu else ":material/warning:",
        color="green" if has_gpu else "orange",
    )

    st.subheader("Queue Zone (Copper)")
    q_x1 = st.number_input("X1", value=50, min_value=0, max_value=959, key="q_x1")
    q_y1 = st.number_input("Y1", value=200, min_value=0, max_value=539, key="q_y1")
    q_x2 = st.number_input("X2", value=400, min_value=1, max_value=960, key="q_x2")
    q_y2 = st.number_input("Y2", value=500, min_value=1, max_value=540, key="q_y2")

    st.subheader("Service Zone (Teal)")
    s_x1 = st.number_input("X1", value=420, min_value=0, max_value=959, key="s_x1")
    s_y1 = st.number_input("Y1", value=200, min_value=0, max_value=539, key="s_y1")
    s_x2 = st.number_input("X2", value=650, min_value=1, max_value=960, key="s_x2")
    s_y2 = st.number_input("Y2", value=450, min_value=1, max_value=540, key="s_y2")

    st.subheader("Detection & Performance")
    conf_threshold = st.slider("Confidence", 0.1, 1.0, 0.45, 0.05, key="conf")
    min_height = st.slider("Min height (px)", 20, 200, 60, key="min_h")
    frame_skip = st.slider(
        "Detect every N frames", 1, 6, 3,
        help="Higher = less CPU load", key="fskip"
    )
    process_width = st.select_slider("YOLO width", [256, 320, 416, 480], value=320, key="pw")

    if st.button("Reset Tracker State", type="primary"):
        for key in ("monitor", "processor"):
            st.session_state.pop(key, None)
        st.rerun()

# Initialize or update session monitor
if "monitor" not in st.session_state:
    cfg = Config()
    cfg.queue_roi = [q_x1, q_y1, q_x2, q_y2]
    cfg.service_roi = [s_x1, s_y1, s_x2, s_y2]
    cfg.yolo_conf = conf_threshold
    cfg.min_person_height = min_height
    cfg.frame_skip = frame_skip
    cfg.process_width = process_width
    st.session_state.monitor = SmartQueueMonitor(cfg)
    st.session_state.processor = QueueVideoProcessor(st.session_state.monitor)

monitor = st.session_state.monitor
processor = st.session_state.processor

monitor.cfg.queue_roi = monitor.fix_roi([q_x1, q_y1, q_x2, q_y2])
monitor.cfg.service_roi = monitor.fix_roi([s_x1, s_y1, s_x2, s_y2])
monitor.cfg.yolo_conf = conf_threshold
monitor.cfg.min_person_height = min_height
monitor.cfg.frame_skip = frame_skip
monitor.cfg.process_width = process_width


# -----------------------------------------------------------------------------
# MODE 1: LIVE WEBRTC WEBCAM
# -----------------------------------------------------------------------------
if app_mode == "📹 Live Webcam (WebRTC)":
    video_col, stats_col = st.columns([3, 1], gap="medium")

    with video_col:
        rtc_config = get_rtc_config()
        webrtc_ctx = webrtc_streamer(
            key="queue-monitor-webrtc",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            video_processor_factory=lambda: processor,
            media_stream_constraints={
                "video": {"width": {"ideal": 960}, "height": {"ideal": 540},
                          "frameRate": {"ideal": 30, "min": 15}},
                "audio": False,
            },
            async_processing=True,
        )
        st.session_state.webrtc_ctx = webrtc_ctx

    def live_stats():
        proc = st.session_state.get("processor")
        if proc is None:
            return
        s = proc.get_stats()

        cols = st.columns(4)
        cols[0].metric("Queue", s["queue"])
        cols[1].metric("In Service", s["service"])
        cols[2].metric("Served", s["served"])
        cols[3].metric("Predicted Wait", f"{int(s['wait'])} s")

        status_colors = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}
        status_icons = {"LOW": ":material/check_circle:", "MEDIUM": ":material/schedule:", "HIGH": ":material/warning:"}
        show_badge(
            f"Queue Status: {s['status']}",
            icon=status_icons.get(s["status"], ":material/help:"),
            color=status_colors.get(s["status"], "grey"),
        )

        if s["proc_ms"] > 0:
            st.caption(f"Inference: {s['proc_ms']:.0f} ms per processed frame")

        ctx = st.session_state.get("webrtc_ctx")
        if ctx is not None and ctx.state.playing:
            st.success("WebRTC connected — live streaming")
        else:
            st.info("Press **START** on the video player above to activate your camera feed.")

    with stats_col:
        st.subheader("📊 Live Metrics")
        if hasattr(st, "fragment"):
            st.fragment(run_every=1)(live_stats)()
        else:
            live_stats()


# -----------------------------------------------------------------------------
# MODE 2: DEMO / UPLOADED VIDEO
# -----------------------------------------------------------------------------
elif app_mode == "🎬 Demo / Uploaded Video":
    st.info("Run the queue monitoring pipeline on a video file. Great for testing on Streamlit Cloud without camera permissions!")

    v_choice = st.radio("Choose video:", ["Preloaded Demo (sample.mp4)", "Upload MP4 Video"], horizontal=True)
    video_source_path = None

    if v_choice == "Preloaded Demo (sample.mp4)":
        if os.path.exists("sample.mp4"):
            video_source_path = "sample.mp4"
        else:
            st.warning("sample.mp4 was not found in the repository root.")
    else:
        uploaded_file = st.file_uploader("Upload an MP4 or AVI video", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            video_source_path = tfile.name

    if video_source_path:
        col_run, col_stop = st.columns([1, 1])
        start_processing = col_run.button("▶ Run Video", type="primary")
        stop_processing = col_stop.button("⏹ Stop")

        frame_placeholder = st.empty()
        metrics_placeholder = st.empty()

        if start_processing:
            cap = cv2.VideoCapture(video_source_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            delay = 1.0 / fps

            while cap.isOpened() and not stop_processing:
                ret, frame = cap.read()
                if not ret:
                    break

                processed_bgr = monitor.process_frame(frame)
                processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(processed_rgb, use_container_width=True)

                s = monitor.get_stats()
                with metrics_placeholder.container():
                    m_cols = st.columns(4)
                    m_cols[0].metric("Queue", s["queue"])
                    m_cols[1].metric("In Service", s["service"])
                    m_cols[2].metric("Served", s["served"])
                    m_cols[3].metric("Predicted Wait", f"{int(s['wait'])} s")
                    show_badge(f"Queue Status: {s['status']}")

                time.sleep(delay * 0.5)

            cap.release()
            st.success("Video playback complete.")


# -----------------------------------------------------------------------------
# MODE 3: CAMERA SNAPSHOT
# -----------------------------------------------------------------------------
elif app_mode == "📸 Camera Snapshot":
    st.info("Take a photo with your webcam to analyze queue size instantly.")
    img_buffer = st.camera_input("Take a photo")

    if img_buffer is not None:
        bytes_data = img_buffer.getvalue()
        frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        processed_bgr = monitor.process_frame(frame)
        processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)

        st.image(processed_rgb, use_container_width=True)
        s = monitor.get_stats()

        cols = st.columns(4)
        cols[0].metric("Queue", s["queue"])
        cols[1].metric("In Service", s["service"])
        cols[2].metric("Served", s["served"])
        cols[3].metric("Predicted Wait", f"{int(s['wait'])} s")
        show_badge(f"Queue Status: {s['status']}")


# ---- Zone Legend & Information ----
with st.expander("ℹ️ Zone Legend & Architecture", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div style="display:flex;align-items:center;gap:8px;padding:8px;">
                <div style="width:16px;height:16px;background:#FF4D00;border-radius:2px;"></div>
                <span style="font-family:'JetBrains Mono',monospace;font-size:13px;">Queue Zone (Copper)</span>
            </div>
            """, unsafe_allow_html=True)
    with c2:
        st.markdown(
            """
            <div style="display:flex;align-items:center;gap:8px;padding:8px;">
                <div style="width:16px;height:16px;background:#00C8C8;border-radius:2px;"></div>
                <span style="font-family:'JetBrains Mono',monospace;font-size:13px;">Service Zone (Teal)</span>
            </div>
            """, unsafe_allow_html=True)
    with c3:
        st.markdown(
            """
            <div style="display:flex;align-items:center;gap:8px;padding:8px;">
                <div style="width:16px;height:16px;background:#4AD04A;border-radius:2px;"></div>
                <span style="font-family:'JetBrains Mono',monospace;font-size:13px;">Tracked Person</span>
            </div>
            """, unsafe_allow_html=True)

st.divider()
st.caption("Streamlit · YOLOv8 · DeepSORT · OpenCV")
