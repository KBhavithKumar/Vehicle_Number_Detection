import streamlit as st
from inference_sdk import InferenceHTTPClient
import cv2
import easyocr
import numpy as np
from PIL import Image
import os

# -----------------------------
# Roboflow Client
# -----------------------------
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="dmqPKPR96l6c42lWNaG8"
)

reader = easyocr.Reader(['en'])

st.title("🚗 AI License Plate Detection")

uploaded_file = st.file_uploader(
    "Upload Vehicle Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    # save temporarily
    image_path = "temp.jpg"
    image.save(image_path)

    st.write("Detecting plate...")

    result = client.run_workflow(
        workspace_name="kokkulas-workspace",
        workflow_id="find-complete-vehicle-number-plates",
        images={"image": image_path},
        use_cache=True
    )

    pred = result[0]["predictions"]["predictions"][0]

    x = int(pred["x"])
    y = int(pred["y"])
    w = int(pred["width"])
    h = int(pred["height"])

    x1 = int(x - w/2)
    y1 = int(y - h/2)
    x2 = int(x + w/2)
    y2 = int(y + h/2)

    img = cv2.imread(image_path)

    # draw bounding box
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

    st.image(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),
             caption="Detected Plate")

    plate = img[y1:y2, x1:x2]

    st.image(cv2.cvtColor(plate,cv2.COLOR_BGR2RGB),
             caption="Cropped Plate")

    ocr_result = reader.readtext(plate)

    if len(ocr_result) > 0:
        plate_text = ocr_result[0][1]
        st.success(f"Detected Plate Number: {plate_text}")
    else:
        st.warning("Could not read plate text")