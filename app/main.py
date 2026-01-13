from fastapi import FastAPI, File, UploadFile
from app.model import model, predict_image
from app.gradcam import generate_gradcam
from app.utils import read_image
from app.schemas import PredictionResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="Chest X-ray AI Demo",
    description="Decision-support tool for chest X-ray interpretation",
    version="1.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # для демо допустимо
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# app.mount("/static", StaticFiles(directory="app/static"), name="static")
#
#
# @app.get("/")
# def serve_frontend():
#     return FileResponse("app/static/index.html")


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    explain: bool = True
):
    contents = await file.read()
    img = read_image(contents)

    result = predict_image(img)

    gradcam_img = None
    if explain and (result["confidence"] < 0.6 or result["predicted_class"] != "negative"):
        gradcam_img = generate_gradcam(model, img)

    return PredictionResponse(
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        probabilities=result["probs"],
        gradcam=gradcam_img
    )
