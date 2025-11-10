import tensorflow as tf
import librosa
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import io
import soundfile as sf

app = FastAPI()

# Load your trained model
model = tf.keras.models.load_model('model_music.keras')

# allow local dev origins (adjust if you host UI elsewhere)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000", "file://", "*"],  # use "*" for quick dev (not for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Class labels
classes = ['pop', 'metal', 'classical', 'jazz', 'rock', 'disco', 'reggae', 'blues', 'hiphop', 'country']

# -------------------- Helper Function --------------------
def predict_genre_from_file(file: UploadFile, model, classes, target_shape=(150, 150)):
    # Step 1: Read uploaded file into memory
    contents = file.file.read()

    # Step 2: Convert file bytes to numpy array
    audio_data, sample_rate = sf.read(io.BytesIO(contents))

    # Step 3: Define chunk parameters
    chunk_duration = 4  # seconds
    overlap_duration = 2
    chunk_samples = int(chunk_duration * sample_rate)
    overlap_samples = int(overlap_duration * sample_rate)

    # Step 4: Create overlapping chunks
    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
    data = []

    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

        # Step 5: Convert to Mel Spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel_spectrogram = tf.image.resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram.numpy())

    # Step 6: Predict genre
    data = np.array(data)
    preds = model.predict(data, verbose=0)
    mean_probs = np.mean(preds, axis=0)
    predicted_index = np.argmax(mean_probs)
    predicted_genre = classes[predicted_index]

    return predicted_genre

# -------------------- API Endpoint --------------------
@app.post("/predict_genre/")
async def get_genre(audio_data: UploadFile = File(...)):
    genre = predict_genre_from_file(audio_data, model, classes)
    return JSONResponse(content={"predicted_genre": genre})

# -------------------- Run the App --------------------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
