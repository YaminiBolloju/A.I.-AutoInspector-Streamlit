import streamlit as st
from PIL import Image
import torch
import gdown
import os
import torchvision.transforms as transforms

# Set up page
st.set_page_config(page_title="AutoInspector AI", layout="centered")
st.title("🚗 🤖 A.I. AutoInspector - Vehicle Damage Detector")

# Define model path and download if not exists
@st.cache_resource
def load_model():
    model_path = "models/damage_segmentation_model.pth"
    file_id = "1RSwVcjB9aidV8hETvYYJvrA9uspg42yc"
    url = f"https://drive.google.com/uc?id={file_id}"

    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists(model_path):
        gdown.download(url, model_path, quiet=False)

    model = torch.load(model_path, map_location=torch.device("cpu"))
    model.eval()
    return model

model = load_model()

# Image upload
uploaded_file = st.file_uploader("Upload a car image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Predict
    with st.spinner("Analyzing damage..."):
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()

        if prediction == 0:
            damage_type = "No visible damage"
            cost = "$0"
            advice = "No repairs needed."
        elif prediction == 1:
            damage_type = "Front Bumper Dent"
            cost = "$450 - $600"
            advice = "Visit a certified collision repair center."
        elif prediction == 2:
            damage_type = "Scratches on Side Panel"
            cost = "$200 - $400"
            advice = "Consider a repaint or buffing job."
        else:
            damage_type = "Unknown"
            cost = "TBD"
            advice = "Further inspection required."

    st.success(f"✅ Damage detected: **{damage_type}**")
    st.write(f"**Repair Recommendation:** {advice}")
    st.write(f"**Estimated Cost:** {cost}")
