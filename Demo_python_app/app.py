import streamlit as st
import tensorflow as tf
import numpy as np
import librosa

# Load the model
model = tf.saved_model.load('./')

# Define the classes
classes = [  "Asthma" ,  "Asthma and lung fibrosis" ,  "BRON" ,  "COPD" ,  "Heart Failure" ,  "Heart Failure + COPD" ,  "Heart Failure + Lung Fibrosis" ,  "Lung Fibrosis" ,  "N" ,  "Plueral Effusion" ,  "pneumonia" ,  ]

def main():
    st.title("Audio Disease Classification")

    # File Upload Widget
    uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3"])

    if uploaded_file is not None:
        # Process the uploaded file
        waveform, sr = load_audio(uploaded_file)

        if waveform is not None:
            st.audio(waveform, format="audio/wav", sample_rate=sr)

            if st.button("Predict Disease"):
                class_prediction = predict_disease(waveform)
                st.write(f"Predicted Disease: {classes[class_prediction]}")

def load_audio(uploaded_file):
    waveform, sr = None, None

    if uploaded_file is not None:
        # Load the audio file
        waveform, sr = librosa.load(uploaded_file, sr=16000)

        if waveform.shape[0] % 16000 != 0:
            waveform = np.concatenate([waveform, np.zeros(16000)])

    return waveform, sr

def predict_disease(waveform):
    inp = tf.constant(np.array([waveform]), dtype='float32')
    class_scores = model(inp)[0].numpy()
    return class_scores.argmax()

if __name__ == '__main__':
    main()
