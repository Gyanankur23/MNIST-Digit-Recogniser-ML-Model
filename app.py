import streamlit as st
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression

@st.cache_resource
def get_model():
    # Load digits dataset (8x8 images)
    digits = load_digits()
    X, y = digits.data / 16.0, digits.target
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    return clf, digits

model, digits = get_model()

st.title("Basic Digit Classifier (No joblib)")
st.write("This demo uses scikit-learn's built-in digits dataset (8×8 grayscale images).")

index = st.slider("Pick a sample index", 0, len(digits.images)-1, 0)
img = digits.images[index]
st.image(img, caption=f"Sample digit (label: {digits.target[index]})", width=150)

if st.button("Predict"):
    arr = digits.data[index].reshape(1, -1) / 16.0
    pred = model.predict(arr)[0]
    st.success(f"Prediction: {pred}")