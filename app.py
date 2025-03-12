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

# Model path
model_path = os.path.join(os.path.dirname(__file__), "model", "lung_cancer_model.h5")
print(f"🔍 Model path: {model_path}")

# Load model only when needed
model = None

def get_model():
    """Load model dynamically to prevent memory issues."""
    global model
    if model is None:
        try:
            print("⚡ Loading model...")
            model = tf.keras.models.load_model(model_path)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            model = None
    return model

# Class labels
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
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flask API is running!"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """Handle image upload and return prediction."""
    if "file" not in request.files:
        print("❌ No file uploaded!")
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    print(f"📂 Received file: {file.filename}")

    model = get_model()
    if model is None:
        return jsonify({"error": "Model failed to load"}), 500

    try:
        processed_image = preprocess_image(io.BytesIO(file.read()))

        print("⚡ Making a prediction...")
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction)
        confidence = float(np.max(prediction) * 100)

        print(f"✅ Prediction: {class_labels[predicted_class]}, Confidence: {confidence:.2f}%")
        return jsonify({"prediction": class_labels[predicted_class], "confidence": f"{confidence:.2f}%"})
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
