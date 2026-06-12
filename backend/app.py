import os
import warnings
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import requests
import json
import base64
from dotenv import load_dotenv
import onnxruntime as ort

# Suppress unnecessary logs
warnings.filterwarnings('ignore')

load_dotenv()

app = Flask(__name__)

# 1. CORS Configuration - allow port 4001
CORS(app, 
     origins=["http://localhost:4001", "http://localhost:5173", "http://localhost:5174"],
     supports_credentials=True,
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"]
)

# ==========================================
# MODEL LOADING (ONNX)
# ==========================================
# Check both Docker path and local path for the model
docker_model_path = "/app/weights/model.onnx"
local_model_path = os.path.join(os.path.dirname(__file__), "model_weights", "model.onnx")
if os.path.exists(docker_model_path):
    MODEL_PATH = docker_model_path
elif os.path.exists(local_model_path):
    MODEL_PATH = local_model_path
else:
    MODEL_PATH = local_model_path  # Fallback

# Session initialization
session = None
input_name = None
output_name = None

print(f"--- Initializing Local Model: {MODEL_PATH} ---")
try:
    if os.path.exists(MODEL_PATH):
        session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        print("SUCCESS: ONNX model loaded locally!")
    else:
        print(f"WARNING: Model file not found at {MODEL_PATH}.")
except Exception as e:
    print(f"ERROR: Failed to load model: {e}")

# ==========================================
# FALLBACK GRADIO API
# ==========================================
def call_gradio_api(data):
    try:
        print("Executing Fallback Analysis via Gradio API...")
        space_url = "https://matthewrt-people-counting.hf.space/run/predict"
        b64_img = base64.b64encode(data).decode('utf-8')
        payload = {
            "data": [f"data:image/jpeg;base64,{b64_img}"],
            "event_data": None, "fn_index": 0, "session_hash": "user_session_" + base64.b64encode(os.urandom(6)).decode('utf-8')
        }
        response = requests.post(space_url, json=payload, timeout=25)
        if response.status_code == 200:
            result = response.json()
            if 'data' in result and len(result['data']) >= 2:
                return int(float(result['data'][1]))
        return None
    except Exception as e:
        print(f"Gradio API failure: {e}")
        return None

# ==========================================
# MAIN COUNTING FUNCTION
# ==========================================
def count_people(image_data):
    """
    Counts people using local ONNX model with fallback to API
    """
    try:
        if session is not None:
            print("Using local ONNX model for inference")
            # Load image from bytes
            img = Image.open(io.BytesIO(image_data)).convert('RGB')
            
            # Preprocess: Resize to 512x512 (model expects this size)
            img_resized = img.resize((512, 512), Image.Resampling.LANCZOS)
            img_array = np.array(img_resized).astype(np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            
            # Run inference
            outputs = session.run([output_name], {input_name: img_array})
            density_map = outputs[0]
            count = float(np.sum(density_map))
            
            # Round to nearest integer
            return int(round(count))
        else:
            print("Local model not available, using Gradio API")
            return call_gradio_api(image_data)
    except Exception as e:
        print(f"Error in count_people: {e}")
        try:
            return call_gradio_api(image_data)
        except:
            return None

# ==========================================
# FLASK ROUTES
# ==========================================
@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": session is not None,
        "model_path": MODEL_PATH
    })

@app.route("/count", methods=["POST"])
def count_route():
    """Main endpoint to count people from uploaded image"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        # Read the file bytes
        image_data = file.read()
        
        # Get the count
        count = count_people(image_data)
        
        if count is None:
            return jsonify({"error": "Failed to count people"}), 500
        
        return jsonify({
            "count": count,
            "source": "local_model" if session is not None else "gradio_api"
        })
    
    except Exception as e:
        print(f"Error in /count route: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/count-base64", methods=["POST"])
def count_base64_route():
    """Endpoint to count people from base64 encoded image string"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No base64 image provided"}), 400
        
        # Decode base64
        base64_str = data['image']
        # Remove header if present (data:image/jpeg;base64,)
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        
        image_data = base64.b64decode(base64_str)
        
        # Get the count
        count = count_people(image_data)
        
        if count is None:
            return jsonify({"error": "Failed to count people"}), 500
        
        return jsonify({
            "count": count,
            "source": "local_model" if session is not None else "gradio_api"
        })
    
    except Exception as e:
        print(f"Error in /count-base64 route: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 4000))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
