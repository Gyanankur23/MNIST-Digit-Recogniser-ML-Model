import gradio as gr
import numpy as np

# Simple "model": sum of pixels mod 10
def predict_digit(img):
    arr = np.array(img.convert("L").resize((28,28)))
    return int(arr.sum() % 10)

demo = gr.Interface(
    fn=predict_digit,
    inputs=gr.Image(shape=(28,28), image_mode="L", invert_colors=True),
    outputs="label",
    title="Digit Classifier Demo"
)

demo.launch()