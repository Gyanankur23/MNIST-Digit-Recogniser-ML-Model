import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load pre-trained model
@st.cache_resource
def get_model():
    return load_model("mnist_model.h5")

model = get_model()

st.title("MNIST Digit Classifier (Pre-trained)")
st.write("Upload a 28×28 grayscale digit image (PNG/JPG).")

uploaded = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])
if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (28,28), interpolation=cv2.INTER_AREA)
    arr = img_resized.reshape(1,28,28,1).astype("float32")/255.0

    st.image(img_resized, caption="Preprocessed 28×28", width=150, channels="GRAY")

    pred = model.predict(arr)[0]
    digit = int(np.argmax(pred))
    confidence = float(np.max(pred))
    st.success(f"Prediction: {digit} (confidence {confidence:.2f})")
