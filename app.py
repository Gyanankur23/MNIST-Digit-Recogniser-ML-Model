import streamlit as st
import numpy as np

st.title("Digits 0–9 Demo (No external datasets)")

# Create 10 synthetic "digit images" as numpy arrays
digits = {}
for d in range(10):
    # Simple pattern: fill diagonal with digit value * 25
    img = np.zeros((28,28), dtype=np.uint8)
    np.fill_diagonal(img, d*25)
    digits[d] = img

choice = st.selectbox("Pick a digit (0–9)", list(digits.keys()))
img = digits[choice]

st.image(img, caption=f"Synthetic digit {choice}", width=150, channels="GRAY")

if st.button("Predict"):
    # Trivial "prediction": just echo the choice
    st.success(f"Predicted digit: {choice}")
