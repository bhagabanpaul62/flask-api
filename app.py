from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import io
from PIL import Image
import os

app = Flask(__name__)
CORS(app)

# Load the model safely
model_path = os.path.join(os.path.dirname(__file__), "model", "lung_cancer_model.h5")
print(f"Loading model from: {model_path}")

try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define class labels
class_labels = ["lung_aca", "lung_scc", "lung_n"]

def preprocess_image(image_file):
    """Preprocess image for prediction."""
    try:
        img = Image.open(image_file).convert("RGB")
        img = img.resize((150, 150))
        img_array = image.img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

@app.route("/predict", methods=["POST"])
def predict():
    """Handle image upload and return prediction."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if model is None:
        return jsonify({"error": "Model is not loaded"}), 500

    try:
        processed_image = preprocess_image(io.BytesIO(file.read()))
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction)
        confidence = float(np.max(prediction) * 100)

        return jsonify({"prediction": class_labels[predicted_class], "confidence": f"{confidence:.2f}%"})
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
