import os
import io
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

# Optional drawing canvas (installed via requirements)
from streamlit_drawable_canvas import st_canvas

MODEL_PATH = "mnist_cnn.h5"
IMG_SIZE = 28

st.set_page_config(page_title="MNIST Digit Classifier", page_icon="🔢", layout="centered")

# -----------------------------
# Utility functions
# -----------------------------
def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=10):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


@st.cache_resource
def get_or_train_model():
    # If a model file exists, load it
    if os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
            return model
        except Exception:
            pass

    # Train a lightweight model on MNIST (first run only)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Normalize and reshape
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)  # (N, 28, 28, 1)
    x_test = np.expand_dims(x_test, -1)

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    model = build_model()
    # Keep epochs modest for quick training
    model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=3,
        batch_size=128,
        verbose=1
    )
    model.save(MODEL_PATH)
    return model


def preprocess_image(img: Image.Image):
    """
    Convert any PIL image to MNIST-compatible 28x28 grayscale with white background
    and centered digit-like content (simple scaling).
    """
    # Convert to grayscale
    img = img.convert("L")
    # Invert if background is dark (heuristic)
    if np.mean(np.array(img)) < 120:
        img = ImageOps.invert(img)

    # Resize keeping aspect ratio and pad to square
    img.thumbnail((IMG_SIZE, IMG_SIZE))
    # Create white square
    canvas = Image.new("L", (IMG_SIZE, IMG_SIZE), color=255)
    # Center paste
    x = (IMG_SIZE - img.size[0]) // 2
    y = (IMG_SIZE - img.size[1]) // 2
    canvas.paste(img, (x, y))

    arr = np.array(canvas).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=-1)  # (28,28,1)
    arr = np.expand_dims(arr, axis=0)   # (1,28,28,1)
    return canvas, arr


def predict_digit(model, arr):
    preds = model.predict(arr, verbose=0)[0]
    digit = int(np.argmax(preds))
    confidence = float(np.max(preds))
    return digit, confidence, preds


# -----------------------------
# UI
# -----------------------------
st.title("MNIST Digit Classifier")
st.write("Draw a digit (0–9) or upload a digit image. The app will preprocess it to 28×28 grayscale and predict using a CNN.")

with st.spinner("Initializing model (first run may train briefly)..."):
    model = get_or_train_model()

# Tabs for drawing and upload
tab1, tab2 = st.tabs(["Draw a digit", "Upload an image"])

with tab1:
    st.write("Use your mouse/finger to draw a digit. Recommended: thick strokes, high contrast.")
    canvas_result = st_canvas(
        fill_color="#ffffff",
        stroke_width=12,
        stroke_color="#000000",
        background_color="#ffffff",
        height=196,
        width=196,
        drawing_mode="freedraw",
        key="canvas",
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        predict_draw = st.button("Predict from drawing")
    with col2:
        clear_btn = st.button("Clear drawing")

    if clear_btn:
        st.experimental_rerun()

    if predict_draw and canvas_result.image_data is not None:
        # Convert canvas RGBA image to PIL
        img_rgba = Image.fromarray((canvas_result.image_data).astype("uint8"))
        # Convert to RGB, then to grayscale in preprocess
        img = img_rgba.convert("RGB")
        processed_img, arr = preprocess_image(img)
        digit, conf, preds = predict_digit(model, arr)

        st.image(processed_img, caption="Preprocessed (28×28) image", width=196)
        st.success(f"Prediction: {digit} | Confidence: {conf:.3f}")

        st.bar_chart({"confidence": preds}, height=200)

with tab2:
    uploaded = st.file_uploader("Upload a digit image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        try:
            img = Image.open(uploaded)
            st.image(img, caption="Uploaded image", use_column_width=True)
            processed_img, arr = preprocess_image(img)
            digit, conf, preds = predict_digit(model, arr)

            st.image(processed_img, caption="Preprocessed (28×28) image", width=196)
            st.success(f"Prediction: {digit} | Confidence: {conf:.3f}")
            st.bar_chart({"confidence": preds}, height=200)
        except Exception as e:
            st.error(f"Could not process image: {e}")

st.caption("Model: CNN trained on MNIST (auto-trained and cached on first run).")
