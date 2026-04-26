import torch
from torchvision import models

class ModelLoader:
    def __init__(self, model_path):
        self.model = torch.load(model_path, map_location=torch.device("cpu"))
        self.model.eval()

    def get_model(self):
        return self.model