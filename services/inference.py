import torch
import torch.nn.functional as F
from model.model import build_model
import gdown
import json
import os

# Google Drive download
def download_model():
    if not os.path.exists("model/neuralnet.pth"):
        os.makedirs("model", exist_ok=True)
        print("Downloading model from Google Drive...")
        gdown.download(
            id="1FB3IirDTbhp4eR_MQSZYlO9CZHAANEhc",  # Google Drive file ID
            output="model/neuralnet.pth",
            quiet=False
        )
        print("Model downloaded successfully.")

#Load class mapping
with open("model/class_to_idx.json", "r") as f:
    class_to_idx = json.load(f)

idx_to_class = {v: k for k, v in class_to_idx.items()}

# Load model
def load_model():
    download_model()  # downloads only if file doesn't exist
    model = build_model()
    model.load_state_dict(
        torch.load("model/neuralnet.pth", map_location="cpu")
    )
    model.eval()
    return model

# Predict
def predict(model, tensor):
    model.eval()
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)
        label = idx_to_class[pred.item()]
    return label, confidence.item()
