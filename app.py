import streamlit as st
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from tensorflow.keras.models import load_model


# -----------------------------
# App title
# -----------------------------
st.title("Handwritten Digit Recognition")

st.write(
    "Upload an image of a handwritten digit, and the trained CNN model will predict the digit."
)


# -----------------------------
# Model path
# -----------------------------
MODEL_PATH = Path("models") / "cnn_mnist_model.keras"


# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_digit_model():
    if not MODEL_PATH.exists():
        st.error("Model file not found. Please train and save the CNN model first.")
        st.stop()

    model = load_model(MODEL_PATH)
    return model


model = load_digit_model()


# -----------------------------
# Image preprocessing function
# -----------------------------
def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("L")

    image_array = np.array(image)

    if image_array.mean() > 127:
        image_array = 255 - image_array

    _, image_threshold = cv2.threshold(
        image_array,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    coords = cv2.findNonZero(image_threshold)

    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        digit_crop = image_threshold[y:y+h, x:x+w]
    else:
        digit_crop = image_threshold

    height, width = digit_crop.shape

    if height > width:
        padding = (height - width) // 2
        digit_crop = cv2.copyMakeBorder(
            digit_crop,
            0,
            0,
            padding,
            padding,
            cv2.BORDER_CONSTANT,
            value=0
        )
    else:
        padding = (width - height) // 2
        digit_crop = cv2.copyMakeBorder(
            digit_crop,
            padding,
            padding,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=0
        )

    digit_resized = cv2.resize(digit_crop, (20, 20), interpolation=cv2.INTER_AREA)

    digit_padded = cv2.copyMakeBorder(
        digit_resized,
        4,
        4,
        4,
        4,
        cv2.BORDER_CONSTANT,
        value=0
    )

    digit_normalized = digit_padded.astype("float32") / 255.0

    digit_input = digit_normalized.reshape(1, 28, 28, 1)

    return image, digit_padded, digit_input


# -----------------------------
# Upload image
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a handwritten digit image",
    type=["png", "jpg", "jpeg"]
)


# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:
    original_image, processed_image, model_input = preprocess_image(uploaded_file)

    st.subheader("Uploaded Image")
    st.image(original_image, caption="Original uploaded image", width=250)

    st.subheader("Processed Image")
    st.image(processed_image, caption="Image prepared for MNIST model", width=150)

    prediction_probabilities = model.predict(model_input)

    predicted_digit = np.argmax(prediction_probabilities)

    confidence = np.max(prediction_probabilities) * 100

    st.subheader("Prediction Result")

    st.write(f"Predicted digit: **{predicted_digit}**")
    st.write(f"Confidence: **{confidence:.2f}%**")

    st.subheader("Class Probabilities")

    for digit, probability in enumerate(prediction_probabilities[0]):
        st.write(f"Digit {digit}: {probability * 100:.2f}%")