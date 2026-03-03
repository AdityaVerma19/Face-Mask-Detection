from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("mask_detector_model.keras")

# Export in TensorFlow SavedModel format (Keras 3 way)
model.export("saved_model_format")