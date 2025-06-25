import streamlit as st
import numpy as np
import librosa
import tensorflow as tf

# === Labels (same order as model output) ===
LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# === Load model ===
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("emotion_cnn1d_lstm_final.keras")
    return model

model = load_model()

# === Feature extraction 
def extract_features(audio_file, max_pad_len=174):
    try:
        audio, sr = librosa.load(audio_file, duration=4, offset=0.5)

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

        pad_width = max(0, max_pad_len - mfcc.shape[1])
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

        combined = mfcc.T  
        combined = combined[np.newaxis, ...]  

        return combined

    except Exception as e:
        st.error(f"❌ Error extracting features: {e}")
        return None

# === Streamlit UI ===
st.title("🎧 Mars Audio Emotion Classifier")
st.markdown("Upload a `.wav` speech or song file and predict the emotion.")

uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    features = extract_features(uploaded_file)

    if features is not None:
        prediction = model.predict(features)
        pred_idx = np.argmax(prediction)
        pred_label = LABELS[pred_idx]
        confidence = prediction[0][pred_idx] * 100

        st.success(f"**Predicted Emotion:** {pred_label} ({confidence:.2f}% confidence)")

        st.bar_chart({LABELS[i]: float(prediction[0][i]) for i in range(len(LABELS))})
