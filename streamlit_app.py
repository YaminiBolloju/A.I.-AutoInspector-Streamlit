import streamlit as st
from PIL import Image
import time

st.set_page_config(page_title="AutoInspector AI", layout="centered")
st.title("🚗 A.I. AutoInspector - Vehicle Damage Detector")

uploaded_file = st.file_uploader("Upload a car image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing damage..."):
        time.sleep(2)  # simulate processing delay

        # Simulated results
        st.success("✅ Damage detected: **Front Bumper Dent**")
        st.write("**Repair Recommendation:** Visit a certified collision repair center.")
        st.write("**Estimated Cost:** $450 - $600")
