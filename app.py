import streamlit as st
import numpy as np

st.title("Basic Digit Demo (No sklearn)")

# Create a tiny "model": predict digit as sum of pixels mod 10
def simple_predict(arr):
    return int(arr.sum() % 10)

st.write("Upload any grayscale image (PNG/JPG). The app will resize to 28×28 and predict a digit.")

uploaded = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])
if uploaded:
    # Read file bytes into numpy array
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)

    # Fake preprocessing: take first 784 values as 'pixels'
    arr = file_bytes[:784].reshape(28, 28) / 255.0

    st.image(arr, caption="Preprocessed 28×28", width=150, channels="GRAY")

    pred = simple_predict(arr)
    st.success(f"Predicted digit (synthetic): {pred}")