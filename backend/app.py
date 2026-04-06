from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
from datetime import datetime
import os
from huggingface_hub import hf_hub_download

# ═══════════════════════════════════════════════════
# CREATE FLASK APP FIRST!
# ═══════════════════════════════════════════════════
app = Flask(__name__)
CORS(app)

# ═══════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════
HF_REPO_ID = "arjit22/SKIN_CANCER_DETECTION"
MODEL_FILENAME = "resnet50_melanoma_final.keras"

CLASS_LABELS = ["Benign", "Malignant"]
COLORS = {
    "Benign": "#10b981",
    "Malignant": "#ef4444"
}

# ═══════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════
print("=" * 60)
print("SKIN CANCER MODEL INITIALIZATION")
print("=" * 60)

model = None
input_shape = (224, 224)

try:
    # Check local model path
    local_model_path = os.path.join("model", MODEL_FILENAME)
    
    print(f"🔍 Checking for local model at: {local_model_path}")
    print(f"📂 Current working directory: {os.getcwd()}")
    print(f"📂 Does model folder exist? {os.path.exists('model')}")
    print(f"📂 Does model file exist? {os.path.exists(local_model_path)}")
    
    if os.path.exists(local_model_path):
        print(f"✓ Found local model: {local_model_path}")
        model_path = local_model_path
    else:
        print("📥 Local model not found. Attempting download from Hugging Face...")
        print(f"   Repository: {HF_REPO_ID}")
        print(f"   File: {MODEL_FILENAME}")
        
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=MODEL_FILENAME,
            cache_dir="./model_cache"
        )
        print(f"✓ Model downloaded to: {model_path}")

    print("📦 Loading model into memory...")
    model = tf.keras.models.load_model(model_path)
    input_shape = model.input_shape[1:3]

    print("✓ Model loaded successfully!")
    print(f"   Input shape: {input_shape}")
    print(f"   Total parameters: {model.count_params():,}")
    print("=" * 60)

except Exception as e:
    print(f"✗ ERROR loading model: {e}")
    import traceback
    traceback.print_exc()
    print("=" * 60)
    model = None
    input_shape = (224, 224)


# ═══════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════
def preprocess_image(image_bytes):
    """Preprocess skin lesion image for model prediction."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize(input_shape)
        img_array = np.array(image, dtype=np.float32)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {str(e)}")


def get_risk_level(confidence, prediction):
    """Determine risk level based on prediction and confidence."""
    if prediction == "Benign":
        if confidence >= 90:
            return "Low Risk", "The lesion appears benign with high confidence."
        elif confidence >= 70:
            return "Low-Moderate Risk", "The lesion likely benign, but monitor for changes."
        else:
            return "Moderate Risk", "Uncertain classification. Medical consultation recommended."
    else:
        if confidence >= 90:
            return "High Risk", "Strong indication of malignancy. Immediate medical attention required."
        elif confidence >= 70:
            return "Moderate-High Risk", "Possible malignancy detected. Urgent medical consultation needed."
        else:
            return "Moderate Risk", "Inconclusive results. Professional examination necessary."


# ═══════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "model": "Skin Cancer Binary Classifier (ResNet50)",
        "version": "1.0",
        "model_loaded": model is not None,
        "input_shape": f"{input_shape[0]}x{input_shape[1]}",
        "classes": CLASS_LABELS,
        "timestamp": datetime.now().isoformat()
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }), 200


@app.route("/model-info", methods=["GET"])
def model_info():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500
    
    return jsonify({
        "model_name": "Skin Cancer Binary Classifier (ResNet50)",
        "input_shape": str(model.input_shape),
        "output_shape": str(model.output_shape),
        "total_parameters": int(model.count_params()),
        "classes": CLASS_LABELS,
        "input_size": f"{input_shape[0]}x{input_shape[1]}",
        "model_type": "Binary Classification (Skin Lesion Analysis)",
        "huggingface_repo": HF_REPO_ID
    })


@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        if "file" in request.files:
            file = request.files["file"]
            if file.filename == "":
                return jsonify({"error": "No file selected"}), 400
            image_bytes = file.read()
        
        elif request.json and "image_base64" in request.json:
            try:
                image_data = request.json["image_base64"]
                if "," in image_data:
                    image_data = image_data.split(",")[1]
                image_bytes = base64.b64decode(image_data)
            except Exception as e:
                return jsonify({"error": f"Invalid base64 image: {str(e)}"}), 400
        
        else:
            return jsonify({"error": "No image provided"}), 400
        
        if len(image_bytes) > 10 * 1024 * 1024:
            return jsonify({"error": "Image too large. Maximum size is 10MB"}), 400
        
        processed_image = preprocess_image(image_bytes)
        prediction_prob = model.predict(processed_image, verbose=0)[0]
        
        if len(prediction_prob) == 1:
            malignant_prob = float(prediction_prob[0])
            benign_prob = 1.0 - malignant_prob
        else:
            benign_prob = float(prediction_prob[0])
            malignant_prob = float(prediction_prob[1])
        
        if malignant_prob > benign_prob:
            prediction = "Malignant"
            confidence = malignant_prob * 100
        else:
            prediction = "Benign"
            confidence = benign_prob * 100
        
        risk_level, recommendation = get_risk_level(confidence, prediction)
        
        result = {
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "probabilities": {
                "Benign": round(benign_prob * 100, 2),
                "Malignant": round(malignant_prob * 100, 2)
            },
            "risk_level": risk_level,
            "recommendation": recommendation,
            "color": COLORS[prediction],
            "timestamp": datetime.now().isoformat(),
            "disclaimer": "This is an AI prediction tool and NOT a substitute for professional medical diagnosis."
        }
        
        return jsonify(result)
    
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# ═══════════════════════════════════════════════════
# RUN THE APP
# ═══════════════════════════════════════════════════
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)