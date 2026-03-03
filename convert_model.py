from tensorflow.keras.models import load_model

model = load_model("mask_detector_model.keras")

model.save("mask_detector_model.h5")

print("Model saved in H5 format")