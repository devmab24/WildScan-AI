from fastapi import APIRouter, UploadFile, File

router = APIRouter()

def get_routes(inference_service):
    
    @router.get("/")
    def home():
        return {"message": "Animal Classifier API is running"}

    @router.post("/predict")
    async def predict(file: UploadFile = File(...)):
        result = inference_service.run_inference(file)
        return result

    return router