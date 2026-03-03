import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

st.title("🎭 Face Mask Detection (Webcam)")

# Load model
model = load_model("mask_detector_model.keras")

# Load face detector
faceNet = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

image = st.camera_input("Take a picture")

if image is not None:
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
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