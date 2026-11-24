#!/bin/bash
pip install --upgrade setuptools
pip install -r requirements.txt
streamlit run streamlit_webrtc_app.py --server.port $PORT --server.address 0.0.0.0