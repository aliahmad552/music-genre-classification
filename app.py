import tensorflow as tf
import librosa
import numpy as np
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import io
import soundfile as sf

app = FastAPI()

# Setup templates and static folders
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model
model = tf.keras.models.load_model('model_music.keras')

# Class labels
classes = ['pop', 'metal', 'classical', 'jazz', 'rock', 'disco', 'reggae', 'blues', 'hiphop', 'country']


# -------------------- Helper Function --------------------
def predict_genre_from_file(file: UploadFile, model, classes, target_shape=(150, 150)):
    contents = file.file.read()
    audio_data, sample_rate = sf.read(io.BytesIO(contents))

    chunk_duration = 4
    overlap_duration = 2
    chunk_samples = int(chunk_duration * sample_rate)
    overlap_samples = int(overlap_duration * sample_rate)

    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
    data = []

    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel_spectrogram = tf.image.resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram.numpy())

    data = np.array(data)
    preds = model.predict(data, verbose=0)
    mean_probs = np.mean(preds, axis=0)

    predicted_index = np.argmax(mean_probs)
    predicted_genre = classes[predicted_index]
    confidence = float(mean_probs[predicted_index])

    return predicted_genre, confidence


# -------------------- ROUTES --------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict_genre/")
async def get_genre(request: Request, audio_data: UploadFile = File(...)):
    genre, confidence = predict_genre_from_file(audio_data, model, classes)
    return JSONResponse(content={"predicted_genre": genre, "confidence": round(confidence * 100, 2)})


# -------------------- Run --------------------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
