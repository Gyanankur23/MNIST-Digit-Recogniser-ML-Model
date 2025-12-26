import streamlit as st
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
import joblib
from PIL import Image, ImageOps

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

def preprocess_image(img: Image.Image):
    img = img.convert("L")
    img = ImageOps.invert(img)
    img = img.resize((28, 28))
    arr = np.array(img).reshape(1, -1) / 255.0
    return img, arr

st.title("MNIST Digit Classifier (Lightweight)")
model = get_model()

uploaded = st.file_uploader("Upload a digit image (PNG/JPG)", type=["png","jpg","jpeg"])
if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", use_column_width=True)
    processed_img, arr = preprocess_image(img)
    st.image(processed_img, caption="Preprocessed 28x28", width=150)
    pred = model.predict(arr)[0]
    st.success(f"Prediction: {pred}")
