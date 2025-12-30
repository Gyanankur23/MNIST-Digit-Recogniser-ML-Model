import streamlit as st
import numpy as np
import struct
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# -----------------------------
# Load MNIST ubyte files
# -----------------------------
def load_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows, cols)

def load_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

# -----------------------------
# App UI
# -----------------------------
st.title("MNIST Logistic Regression Dashboard")

# Upload files
train_images_file = st.file_uploader("Upload train-images-idx3-ubyte", type=["ubyte"])
train_labels_file = st.file_uploader("Upload train-labels-idx1-ubyte", type=["ubyte"])
test_images_file = st.file_uploader("Upload t10k-images-idx3-ubyte", type=["ubyte"])
test_labels_file = st.file_uploader("Upload t10k-labels-idx1-ubyte", type=["ubyte"])

if train_images_file and train_labels_file and test_images_file and test_labels_file:
    # Load data
    X_train = load_images(train_images_file)
    y_train = load_labels(train_labels_file)
    X_test = load_images(test_images_file)
    y_test = load_labels(test_labels_file)

    # Flatten
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # Train model
    st.write("Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_flat, y_train)

    y_pred = model.predict(X_test_flat)
    acc = accuracy_score(y_test, y_pred)
    st.success(f"Test Accuracy: {acc:.2f}")

    # Interactive prediction
    idx = st.slider("Select test image index", 0, len(X_test)-1, 0)
    fig, ax = plt.subplots()
    ax.imshow(X_test[idx], cmap="gray")
    ax.axis("off")
    st.pyplot(fig)

    st.write(f"True Label: {y_test[idx]}")
    st.write(f"Predicted Label: {model.predict([X_test_flat[idx]])[0]}")