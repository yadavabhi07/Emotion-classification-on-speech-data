import numpy as np
import librosa
import tensorflow as tf
import sys

# === Emotion labels 
LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

def load_model(model_path="emotion_cnn1d_lstm_final.keras"):
    """Load the trained Keras model"""
    try:
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully.")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

def extract_features(file_path, max_pad_len=174):
    """Extract MFCC features from an audio file"""
    try:
        audio, sr = librosa.load(file_path, duration=4, offset=0.5)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        pad_width = max(0, max_pad_len - mfcc.shape[1])
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        combined = mfcc.T  
        combined = combined[np.newaxis, ...]  
        return combined
    except Exception as e:
        print(f"❌ Error extracting features: {e}")
        sys.exit(1)

def predict_emotion(model, features):
    """Predict emotion from features"""
    prediction = model.predict(features)
    pred_idx = np.argmax(prediction)
    pred_label = LABELS[pred_idx]
    confidence = prediction[0][pred_idx] * 100
    print(f"\n🎯 Predicted Emotion: {pred_label} ({confidence:.2f}% confidence)\n")

    print("📊 Probabilities for each emotion:")
    for label, prob in zip(LABELS, prediction[0]):
        print(f"{label:10}: {prob:.4f}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python test_model.py path_to_audio.wav")
        sys.exit(1)

    audio_path = sys.argv[1]
    print(f"🔍 Processing: {audio_path}")

    model = load_model()
    features = extract_features(audio_path)
    predict_emotion(model, features)

if __name__ == "__main__":
    main()
