from fastapi import FastAPI
from app.infrastructure.model_loader import ModelLoader
from app.domain.model_service import ModelService
from app.application.inference import InferenceService
from app.api.routes import get_routes

app = FastAPI()

# Load model
loader = ModelLoader("model/animal_model.pth")
model = loader.get_model()

# Initialize layers
model_service = ModelService(model)
inference_service = InferenceService(model_service)

# Register routes
app.include_router(get_routes(inference_service))