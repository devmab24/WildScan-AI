import streamlit as st
from PIL import Image
import torch
from torchvision import transforms

# -----------------------------
# UI SETUP
# -----------------------------
st.set_page_config(page_title="WildScan AI", layout="centered")

st.title("🐾 WildScan AI")
st.write("Upload an animal image to classify it into 8 categories.")

# -----------------------------
# SAFE MODEL LOADING
# -----------------------------
@st.cache_resource
def load_model():
    try:
        model = torch.load("model/animal_model.pth", map_location="cpu")
        model.eval()
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

model = load_model()

# -----------------------------
# CLASS LABELS
# -----------------------------
classes = [
    "cat", "dog", "elephant", "lion",
    "tiger", "zebra", "giraffe", "horse"
]

# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def preprocess(image):
    return transform(image).unsqueeze(0)

# -----------------------------
# UI INPUT
# -----------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if model is None:
        st.warning("Model not loaded. Please check file path.")
    else:
        tensor = preprocess(image)

        with torch.no_grad():
            output = model(tensor)
            _, pred = torch.max(output, 1)
            result = classes[pred.item()]

        st.success(f"Prediction: **{result}**")
        
# import streamlit as st
# from PIL import Image

# from model_loader import load_model
# from utils import preprocess
# from inference import predict

# # Load model
# model = load_model("model/neuralnet.pth")

# classes = [
#     "bird", "monkey_prosimian", "leopard", "hog",
#     "civet_genet", "antelope_duiker", "blank", "rodent"
# ]

# st.title("🐾 WildScan AI - Animal Classifier")

# uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# if uploaded_file:
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     tensor = preprocess(image)
#     result = predict(model, tensor, classes)

#     st.success(f"Prediction: {result}")