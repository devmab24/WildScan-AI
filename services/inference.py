import torch
import torch.nn.functional as F
from model.model import build_model

import json
import torch

# Load class mapping
with open("model/class_to_idx.json", "r") as f:
    class_to_idx = json.load(f)

# Convert to index → class
idx_to_class = {v: k for k, v in class_to_idx.items()}

def load_model():
    model = build_model()
    model.load_state_dict(
        torch.load("model/neuralnet.pth", map_location="cpu")
    )
    model.eval()
    return model

def predict(model, tensor):
    model.eval()

    with torch.no_grad():
        output = model(tensor)

        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)

        label = idx_to_class[pred.item()]

    return label, confidence.item()
# #Prediction function
# def predict(model, tensor, idx_to_class):
#     model.eval()

#     with torch.no_grad():
#         output = model(tensor)

#         probs = torch.softmax(output, dim=1)
#         confidence, pred = torch.max(probs, 1)

#         label = idx_to_class[pred.item()]

#     return label, confidence.item()
# def predict(model, tensor, classes):
#     with torch.no_grad():
#         output = model(tensor)
#         probs = F.softmax(output, dim=1)

#         top_prob, top_class = torch.max(probs, 1)

#     return classes[top_class.item()], top_prob.item()