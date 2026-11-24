#!/bin/bash
pip install --force-reinstall setuptools
pip install -r requirements.txt
streamlit run streamlit_webrtc_app.py --server.port $PORT --server.address 0.0.0.0
