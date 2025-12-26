import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegression

@st.cache_resource
def get_model():
    # Create a tiny synthetic dataset: "digits" are just sums of pixel blocks
    # X: 100 samples of 8x8 random "images"
    X = np.random.randint(0, 16, (100, 64))
    # y: labels are just (sum of pixels mod 10)
    y = (X.sum(axis=1) % 10)
    clf = LogisticRegression(max_iter=500)
    clf.fit(X, y)
    return clf

model = get_model()

st.title("Synthetic Digit Classifier")
st.write("Demo without external datasets. Uses synthetic data for training.")

# Let user upload an image (optional)
uploaded = st.file_uploader("Upload a grayscale digit image (PNG/JPG)", type=["png","jpg","jpeg"])

if uploaded:
    # Convert uploaded file to numpy array
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    # Just take first 64 values as a fake 'digit'
    arr = file_bytes[:64].reshape(1, -1) % 16
    pred = model.predict(arr)[0]
    st.success(f"Predicted digit (synthetic): {pred}")
else:
    st.info("Upload any image file to see a synthetic prediction.")