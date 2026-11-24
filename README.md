# Smart Queue Monitor ⏱️

An intelligent real-time queue detection and waiting-time estimation system powered by **YOLOv8**, **DeepSORT**, and **Streamlit**.

---

## 🚀 Features

- **Live WebRTC Video Stream**: Direct real-time browser webcam processing with peer-to-peer WebRTC (`streamlit-webrtc`).
- **Demo / Recorded Video Processing**: Upload videos or test with preloaded `sample.mp4` for offline or cloud testing without camera permissions.
- **Snapshot Analysis**: Instant queue count from single camera photos.
- **Industrial Visual Overlays**:
  - 🟠 **Queue Zone** (Copper)
  - 🔵 **Service Zone** (Teal)
  - 🟢 **Tracked Persons** with persistent DeepSORT IDs
- **Predictive Metrics**: Live estimation of wait times:
  $$\text{Predicted Wait} = \frac{\text{Queue Count} \times \text{Avg Service Time}}{\max(\text{Active Servers}, 1)}$$

---

## ☁️ Deploying to Streamlit Community Cloud

1. **Fork or push this repository** to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io/).
3. Click **"New App"** and select this repository.
4. Set the configuration:
   - **Repository:** `<your-repo>`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
5. Click **"Deploy!"**.

### WebRTC on Cloud Networks (Optional TURN Server)
By default, free Google STUN servers are configured. However, if your cloud network or client is behind strict firewalls/NATs, add your TURN credentials under **App Settings > Secrets**:

```toml
TURN_URL = "turn:global.turn.twilio.com:3443?transport=tcp"
TURN_USERNAME = "your_twilio_sid"
TURN_PASSWORD = "your_twilio_auth_token"
```

---

## 💻 Local Setup

1. **Clone repository**:
   ```bash
   git clone https://github.com/your-username/WTP-main.git
   cd WTP-main
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Streamlit application**:
   ```bash
   streamlit run streamlit_app.py
   ```
   *(Or run `streamlit run streamlit_webrtc_app.py` for WebRTC-only mode)*
