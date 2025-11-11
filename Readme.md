# 🎵 Music Genre Classification Project

## Overview

Music is an essential part of human culture, but automatically classifying songs into genres is a challenging problem for computers. With the explosion of digital music libraries, manual tagging is not scalable. This project aims to build an AI system that can accurately predict the genre of a given music clip. By leveraging deep learning, specifically **CNN** and **CRNN** models, we can extract both spatial and temporal features from audio spectrograms and classify songs into ten genres: pop, metal, classical, jazz, rock, disco, reggae, blues, hiphop, and country. This project demonstrates the full AI pipeline: from dataset collection and preprocessing to model training, deployment, and containerization.

---

## Dataset

We used the **GTZAN Music Genre Dataset**, which contains 1000 audio tracks, each 30 seconds long, distributed across 10 genres (100 clips per genre). The dataset provides a diverse set of music styles and is widely used for benchmarking music genre classification systems.

- **Number of samples:** 1000  
- **Genres:** pop, metal, classical, jazz, rock, disco, reggae, blues, hiphop, country  
- **Format:** WAV audio files, 22050 Hz sampling rate

---

## Preprocessing Techniques

Before feeding audio files to the models, several preprocessing steps were applied:

1. **Loading audio files** using `librosa`.
2. **Chunking audio clips** into overlapping segments (4 seconds long, 2 seconds overlap) to increase dataset size and handle long audio sequences.
3. **Converting to Mel Spectrograms** to extract meaningful frequency features.
4. **Resizing spectrograms** for CNN input (150x150 for CNN, original shape for CRNN).
5. **Normalization and reshaping** to match model input dimensions.

These preprocessing steps help the model capture both temporal and spectral patterns in the audio signals.

---

## Model Experiments

We trained multiple deep learning models:

1. **CNN Models**
   - Standard 2D Convolutional layers with pooling and dropout.
   - Experimented with different filter sizes and number of layers.

2. **CRNN Model**
   - CNN layers to extract spatial features from spectrograms.
   - Bidirectional LSTM layers to capture temporal dependencies.
   - Attention mechanism added to focus on important parts of the audio.

### Best Performance

- **CNN Accuracy:** 97% on the validation set
- **CRNN Accuracy:** Slightly better temporal understanding, used for further experimentation

---

## Saving the Model

The best performing CNN model was saved as:

