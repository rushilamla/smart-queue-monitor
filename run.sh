#!/bin/bash
pip install "setuptools<=69.5.1"
pip install -r requirements.txt
streamlit run streamlit_app.py --server.port ${PORT:-8501} --server.address 0.0.0.0
