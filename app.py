from tensorflow as tf
import librosa
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import uvicorn

app = FastAPI()

class AudioData(BaseModel):
    file_path: str

model = tf.keras.models.load_model('model_music.keras')

classes = ['pop','metal','classical','jazz','rock','disco','reggae','blues','hiphop','country']

