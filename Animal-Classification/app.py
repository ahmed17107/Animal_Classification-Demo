import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Class names in training order
CLASS_NAMES = [
    "butterfly",
    "cat",
    "chicken",
    "cow",
    "dog",
    "Elephant",
    "Horse",
    "Sheep",
    "spider",
    "squirrel"
]

# Load model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "animals10_trial_lr_0_0003.keras"
    )

model = load_model()

st.title("🐾 Animal Classification")
st.write("Upload an animal image and let our CNN classify it!")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", width=350)

    # Preprocessing: resize and normalize
    resized_image = image.resize((128, 128))
    image_array = np.array(resized_image, dtype=np.float32)
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    predictions = model.predict(image_array, verbose=0)

    predicted_index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = predictions[0][predicted_index] * 100

    st.success(f"Predicted Animal: {predicted_class}")
    st.metric("Confidence", f"{confidence:.2f}%")

    st.subheader("All Class Predictions")

    for i, name in enumerate(CLASS_NAMES):
        st.write(f"{name}: {predictions[0][i] * 100:.2f}%")
