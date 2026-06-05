import os
import warnings
import numpy as np
import cv2
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import requests
import json
import base64
from torchvision import transforms
from dotenv import load_dotenv

# Suppress unnecessary logs
warnings.filterwarnings('ignore')

load_dotenv()

app = Flask(__name__)

# 1. CORS Configuration
ENV = os.getenv("APP_ENV", "development")
ALLOWED_ORIGINS = ["*"] if ENV != "production" else [os.getenv("PRODUCTION_CORS_ORIGINS", "https://shivaaydhondiyal.online")]

CORS(app, 
     origins=ALLOWED_ORIGINS,
     supports_credentials=True,
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"]
)

# ==========================================
# PYTORCH MODEL ARCHITECTURE
# ==========================================
class PyTorchCBAMLayer(nn.Module):
    def __init__(self, channels, reduction=8):
        super(PyTorchCBAMLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels)
        )
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        max_out = self.fc(self.max_pool(x).view(b, c)).view(b, c, 1, 1)
        x = x * self.sigmoid_channel(avg_out + max_out)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.sigmoid_spatial(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))
        return x * sa

class PyTorchDensityModel(nn.Module):
    def __init__(self):
        super(PyTorchDensityModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            PyTorchCBAMLayer(64),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        return self.features(x)

# ==========================================
# LOCAL MODEL LOADING (PYTORCH)
# ==========================================
MODEL_PATH = "/app/weights/best.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

print(f"--- Initializing Local Model: {MODEL_PATH} ---")
model = None
try:
    if os.path.exists(MODEL_PATH):
        # We try to load the full model object first as requested (best_full.pt was 99MB)
        # If that fails, we can fall back to loading state_dict into the architecture
        try:
            model = torch.load(MODEL_PATH, map_location=device)
        except:
            model = PyTorchDensityModel()
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        
        model.eval()
        print("SUCCESS: PyTorch model loaded locally.")
    else:
        print(f"WARNING: Model file not found at {MODEL_PATH}.")
except Exception as e:
    print(f"ERROR: Failed to load model: {e}")

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0).to(device)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "active", "model_loaded": model is not None})

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    img_bytes = file.read()
    
    results = {
        "hf_analysis": {"count": 0, "tier": "Unknown", "detections": []},
        "classification_scores": []
    }

    # 1. Local PyTorch Inference
    if model:
        try:
            with torch.no_grad():
                input_tensor = preprocess_image(img_bytes)
                density_map = model(input_tensor)
                count = float(torch.sum(density_map).item())
                results["hf_analysis"]["count"] = round(count, 1)
                print(f"Local PyTorch Count: {results['hf_analysis']['count']}")
        except Exception as e:
            print(f"Local Inference Error: {e}")

    # 2. API Fallback
    if results["hf_analysis"]["count"] < 0.1 or not model:
        api_count = call_gradio_api(img_bytes)
        if api_count is not None:
            results["hf_analysis"]["count"] = api_count
            print(f"API Fallback Count: {api_count}")

    # Tier classification
    final_count = results["hf_analysis"]["count"]
    if final_count <= 15: tier = "Sparse"
    elif final_count <= 50: tier = "Normal"
    elif final_count <= 150: tier = "Moderate Dense"
    else: tier = "High Dense"
    
    results["hf_analysis"]["tier"] = tier
    results["classification_scores"] = [{"label": tier, "score": 1.0}]

    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
