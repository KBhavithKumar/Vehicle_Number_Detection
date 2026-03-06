import streamlit as st
import cv2
import numpy as np
import requests
from PIL import Image
import pytesseract

# -----------------------------
# Roboflow API settings
# -----------------------------
API_KEY = "dmqPKPR96l6c42lWNaG8"
WORKSPACE = "kokkulas-workspace"
WORKFLOW = "find-complete-vehicle-number-plates"

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🚗 AI License Plate Detection")

uploaded_file = st.file_uploader(
    "Upload a vehicle image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image")

    # Save image temporarily
    image_path = "temp.jpg"
    image.save(image_path)

    st.write("Detecting license plate...")

    # -----------------------------
    # Roboflow API request
    # -----------------------------
    url = f"https://serverless.roboflow.com/{WORKSPACE}/workflows/{WORKFLOW}?api_key={API_KEY}"

    with open(image_path, "rb") as f:
        response = requests.post(url, files={"image": f})

    result = response.json()

    try:
        pred = result[0]["predictions"]["predictions"][0]

        x = int(pred["x"])
        y = int(pred["y"])
        w = int(pred["width"])
        h = int(pred["height"])

        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        img = cv2.imread(image_path)

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        st.image(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            caption="Detected Plate"
        )

        # Crop plate
        plate = img[y1:y2, x1:x2]

        st.image(
            cv2.cvtColor(plate, cv2.COLOR_BGR2RGB),
            caption="Cropped Plate"
        )

        # -----------------------------
        # OCR using pytesseract
        # -----------------------------
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

        text = pytesseract.image_to_string(gray)

        st.success(f"Detected Plate Number: {text}")

    except:
        st.error("No license plate detected in this image.")
