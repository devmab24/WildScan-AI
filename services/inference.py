import torch
import torch.nn.functional as F
from model.model import build_model
import gdown
import json
import os

# ── Google Drive download ──────────────────────────────────────────────────────
def download_model():
    if not os.path.exists("model/neuralnet.pth"):
        os.makedirs("model", exist_ok=True)
        print("Downloading model from Google Drive...")
        gdown.download(
            id="1FB3IirDTbhp4eR_MQSZYlO9CZHAANEhc",  # replace with your actual file ID
            output="model/neuralnet.pth",
            quiet=False
        )
        print("Model downloaded successfully.")

# ── Load class mapping ─────────────────────────────────────────────────────────
with open("model/class_to_idx.json", "r") as f:
    class_to_idx = json.load(f)

idx_to_class = {v: k for k, v in class_to_idx.items()}

# ── Load model ─────────────────────────────────────────────────────────────────
def load_model():
    download_model()  # downloads only if file doesn't exist
    model = build_model()
    model.load_state_dict(
        torch.load("model/neuralnet.pth", map_location="cpu")
    )
    model.eval()
    return model

# ── Predict ────────────────────────────────────────────────────────────────────
def predict(model, tensor):
    model.eval()
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)
        label = idx_to_class[pred.item()]
    return label, confidence.item()

# import torch
# import torch.nn.functional as F
# from model.model import build_model
# import requests
# import json
# import os

# # ── Google Drive download ──────────────────────────────────────────────────────
# def download_model():
#     if not os.path.exists("model/neuralnet.pth"):
#         os.makedirs("model", exist_ok=True)
#         print("Downloading model from Google Drive...")
#         url = url = "https://drive.google.com/uc?export=download&id=1FB3IirDTbhp4eR_MQSZYlO9CZHAANEhc"  # replace this
#         r = requests.get(url, stream=True)
#         with open("model/neuralnet.pth", "wb") as f:
#             for chunk in r.iter_content(chunk_size=8192):
#                 f.write(chunk)
#         print("Model downloaded successfully.")

# # ── Load class mapping ─────────────────────────────────────────────────────────
# with open("model/class_to_idx.json", "r") as f:
#     class_to_idx = json.load(f)

# idx_to_class = {v: k for k, v in class_to_idx.items()}

# # ── Load model ─────────────────────────────────────────────────────────────────
# def load_model():
#     download_model()  # ← downloads only if file doesn't exist
#     model = build_model()
#     model.load_state_dict(
#         torch.load("model/neuralnet.pth", map_location="cpu")
#     )
#     model.eval()
#     return model

# # ── Predict ────────────────────────────────────────────────────────────────────
# def predict(model, tensor):
#     model.eval()
#     with torch.no_grad():
#         output = model(tensor)
#         probs = torch.softmax(output, dim=1)
#         confidence, pred = torch.max(probs, 1)
#         label = idx_to_class[pred.item()]
#     return label, confidence.item()

# # import torch
# # import torch.nn.functional as F
# # from model.model import build_model

# # import json
# # import torch

# # # Load class mapping
# # with open("model/class_to_idx.json", "r") as f:
# #     class_to_idx = json.load(f)

# # # Convert to index → class
# # idx_to_class = {v: k for k, v in class_to_idx.items()}

# # def load_model():
# #     model = build_model()
# #     model.load_state_dict(
# #         torch.load("model/neuralnet.pth", map_location="cpu")
# #     )
# #     model.eval()
# #     return model

# # def predict(model, tensor):
# #     model.eval()

# #     with torch.no_grad():
# #         output = model(tensor)

# #         probs = torch.softmax(output, dim=1)
# #         confidence, pred = torch.max(probs, 1)

# #         label = idx_to_class[pred.item()]

# #     return label, confidence.item()
# # #Prediction function
# # def predict(model, tensor, idx_to_class):
# #     model.eval()

# #     with torch.no_grad():
# #         output = model(tensor)

# #         probs = torch.softmax(output, dim=1)
# #         confidence, pred = torch.max(probs, 1)

# #         label = idx_to_class[pred.item()]

# #     return label, confidence.item()
# # def predict(model, tensor, classes):
# #     with torch.no_grad():
# #         output = model(tensor)
# #         probs = F.softmax(output, dim=1)

# #         top_prob, top_class = torch.max(probs, 1)

# #     return classes[top_class.item()], top_prob.item()