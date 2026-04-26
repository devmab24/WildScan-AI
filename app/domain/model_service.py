import torch

class ModelService:
    def __init__(self, model):
        self.model = model
        self.classes = [
            "cat", "dog", "elephant", "lion",
            "tiger", "zebra", "giraffe", "horse"
        ]

    def predict(self, tensor):
        with torch.no_grad():
            outputs = self.model(tensor)
            _, predicted = torch.max(outputs, 1)

        return self.classes[predicted.item()]