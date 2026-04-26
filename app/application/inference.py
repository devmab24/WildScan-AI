from app.infrastructure.image_utils import preprocess_image
from PIL import Image

class InferenceService:
    def __init__(self, model_service):
        self.model_service = model_service

    def run_inference(self, file):
        image = Image.open(file.file).convert("RGB")
        tensor = preprocess_image(image)
        prediction = self.model_service.predict(tensor)

        return {"prediction": prediction}