import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("digit_model.h5")

st.title("Handwritten Digit Recognition")

uploaded_file = st.file_uploader(
    "Upload Digit Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert('L')

    image = image.resize((28,28))

    img_array = np.array(image)

    img_array = img_array / 255.0

    img_array = img_array.reshape(1,28,28,1)

    prediction = model.predict(img_array)

    digit = np.argmax(prediction)

    st.image(image, caption="Uploaded Image")

    st.success(f"Predicted Digit: {digit}")