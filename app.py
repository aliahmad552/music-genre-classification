# this is the backend code for my web application
from fastapi import FastAPI
from pydantic import BaseModel
# ...existing code...
# this is the backend code for my web application
from fastapi import FastAPI
from pydantic import BaseModel
# ...existing code...

from fastapi import UploadFile, File, HTTPException
from typing import List, Optional
import numpy as np
import joblib
import os
import io
import librosa

app = FastAPI(title="Music Genre Prediction API")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Save your trained pipeline as 'model.joblib'.")
    return joblib.load(MODEL_PATH)

model = None

def ensure_model():
    global model
    if model is None:
        model = load_model()
    return model

def preprocess_features_list(features: List[float]) -> np.ndarray:
    arr = np.array(features, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr

def extract_mfcc_from_bytes(file_bytes: bytes, sr: int = 22050, n_mfcc: int = 40) -> List[float]:
    try:
        audio_buffer = io.BytesIO(file_bytes)
        y, _ = librosa.load(audio_buffer, sr=sr, mono=True)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)  # shape (n_mfcc,)
        return mfcc_mean.tolist()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio processing failed: {e}")

class FeaturesRequest(BaseModel):
    features: List[float]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict_json")
def predict_json(req: FeaturesRequest):
    mdl = ensure_model()
    X = preprocess_features_list(req.features)
    preds = mdl.predict(X).tolist()
    probs = None
    if hasattr(mdl, "predict_proba"):
        try:
            probs = mdl.predict_proba(X).tolist()
        except Exception:
            probs = None
    return {"predictions": preds, "probabilities": probs}

@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):
    content = await file.read()
    mfcc_features = extract_mfcc_from_bytes(content)
    return predict_json(FeaturesRequest(features=mfcc_features))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)