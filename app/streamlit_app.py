import streamlit as st
import requests

st.title("Animal Classifier")

file = st.file_uploader("Upload Image")

if file:
    response = requests.post(
        "http://127.0.0.1:8000/predict",
        files={"file": file}
    )
    st.write(response.json())
