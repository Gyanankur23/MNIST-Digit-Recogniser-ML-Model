import streamlit as st
import numpy as np
import cv2
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
import joblib

MODEL_PATH = "mnist_lr.pkl"

@st.cache_resource
def get_model():
    try:
        return joblib.load(MODEL_PATH)
    except:
        # Download MNIST and train once
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
        X = X / 255.0
        y = y.astype(int)
        clf = LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="multinomial")
        clf.fit(X, y)
        joblib.dump(clf, MODEL_PATH)
        return clf

def preprocess_image(file_bytes):
    # Read uploaded image as grayscale
    file_array = np.asarray(bytearray(file_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(file_array, cv2.IMREAD_GRAYSCALE)

    # Resize to 28x28
    img_resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    # Invert colors if background is dark
    if np.mean(img_resized) < 120:
        img_resized = cv2.bitwise_not(img_resized)

    arr = img_resized.reshape(1, -1) / 255.0
    return img_resized, arr

st.title("MNIST Digit Classifier (No Pillow)")
model = get_model()

uploaded = st.file_uploader("Upload a digit image (PNG/JPG)", type=["png","jpg","jpeg"])
if uploaded:
    img_resized, arr = preprocess_image(uploaded)
    st.image(img_resized, caption="Preprocessed 28x28", width=150, channels="GRAY")
    pred = model.predict(arr)[0]
    st.success(f"Prediction: {pred}")