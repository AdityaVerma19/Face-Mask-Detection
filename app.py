import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import zipfile
import os

st.set_page_config(page_title="Face Mask Detection", layout="centered")

st.title("🎭 Face Mask Detection")
st.write("Detect whether a person is wearing a mask.")

# ----------------------------
# Unzip model if needed
# ----------------------------
if not os.path.exists("saved_model_format"):
    with zipfile.ZipFile("saved_model_format.zip", 'r') as zip_ref:
        zip_ref.extractall()

# ----------------------------
# Load Model
# ----------------------------
model = tf.keras.models.load_model("saved_model_format")

# ----------------------------
# Load Face Detector
# ----------------------------
faceNet = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

# ----------------------------
# Input Method
# ----------------------------
option = st.radio("Choose input method", ["Upload", "Camera"])

if option == "Upload":
    img_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
else:
    img_file = st.camera_input("Take a picture")

# ----------------------------
# Process Image
# ----------------------------
if img_file is not None:

    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    (h, w) = img.shape[:2]

    blob = cv2.dnn.blobFromImage(
        img, 1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )

    faceNet.setInput(blob)
    detections = faceNet.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = img[startY:endY, startX:endX]

            if face.size == 0:
                continue

            face = cv2.resize(face, (224, 224))
            face = face / 255.0
            face = np.expand_dims(face, axis=0)

            prediction = model.predict(face, verbose=0)[0][0]

            if prediction < 0.5:
                label = "Mask"
                confidence_score = (1 - prediction) * 100
                color = (0, 255, 0)
            else:
                label = "No Mask"
                confidence_score = prediction * 100
                color = (0, 0, 255)

            text = f"{label} ({confidence_score:.2f}%)"

            cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)
            cv2.putText(img, text,
                        (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)

    st.image(img, channels="BGR")