import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Load model
model = tf.keras.models.load_model("digit_model.h5")

st.title("Handwritten Digit Recognition System")


def predict_digit(img):

    img = img.convert("L")
    img = img.resize((28, 28))

    img_array = np.array(img)

    # Invert colors for MNIST format
    img_array = 255 - img_array

    img_array = img_array / 255.0

    img_array = img_array.reshape(1, 28, 28, 1)

    prediction = model.predict(img_array)

    digit = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    return digit, confidence



st.header("Upload Digit Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", width=200)

    digit, confidence = predict_digit(image)

    st.success(f"Predicted Digit: {digit}")
    st.info(f"Confidence: {confidence:.2f}%")


st.header("Draw Digit on Canvas")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:

    if st.button("Predict Drawn Digit"):

        img = Image.fromarray(
            canvas_result.image_data.astype("uint8")
        )

        digit, confidence = predict_digit(img)

        st.success(f"Predicted Digit: {digit}")
        st.info(f"Confidence: {confidence:.2f}%")