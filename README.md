# Mars Audio Emotion Classification 🎧

This project implements an end-to-end pipeline for **emotion classification using speech and song audio data**.  
We process audio files using MFCC, chroma, and spectral contrast features, and classify them with a **CNN1D + LSTM deep learning model**.

---

## 🚀 Project Description

The objective is to:
- Accurately identify emotions conveyed in speech or song.
- Build a robust model that meets the following criteria:
  - F1 score > 80%
  - Accuracy per class > 75%
  - Overall accuracy > 80%

We also provide a **Streamlit web app** that allows real-time emotion prediction on uploaded `.wav` files.

---

## 📂 Dataset

The dataset contains speech and song audio files labeled with 8 emotion categories:
- `neutral`
- `calm`
- `happy`
- `sad`
- `angry`
- `fearful`
- `disgust`
- `surprised`

---

## 🛠️ Pre-processing methodology

- Extract **MFCC (40)**, **chroma (12)**, and **spectral contrast (7)** features → combined into 59-dimensional feature vectors.
- Pad/truncate all sequences to `174` time steps.
- Normalize features (mean/std) per file.
- Train/test split: 80% training, 20% validation (stratified).

---

## ⚙️ Model pipeline

- **Conv1D layers** with batch normalization, max pooling, and dropout
- **Bidirectional LSTM**
- Fully connected dense layers with dropout
- Final softmax layer for 8-class classification
- Optimizer: Adam
- Loss: categorical crossentropy
- Metrics: accuracy

---

## ✅ Accuracy metrics

| Metric | Value |
|---------|-------|
| Overall accuracy | > 80% |
| Per-class accuracy | > 75% (all classes) |
| F1 Score | > 80% |
| Confusion matrix | Provided in notebook |

---

## 💻 How to run the notebook

1️⃣ Open and run `Mars_Audio_Classification_Report.ipynb`  
2️⃣ The notebook will:
- Load audio files
- Extract features
- Train and evaluate the model
- Save the trained model (`emotion_cnn1d_lstm_final.keras`)

---

## 🌐 Streamlit web app

We provide `app.py` for a simple Streamlit web interface.

### Run locally:
```bash
streamlit run app.py