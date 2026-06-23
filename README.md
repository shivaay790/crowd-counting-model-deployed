# Crowd Counting Using CBAM-Enhanced CNN Model

## 📸 What is this?

This is an intelligent web application that counts people in images and videos using advanced deep learning. Our model analyzes crowd photos and gives you an accurate count, plus extra features like crowd size classification and risk zone detection!

## 🎯 Key Features

- **Accurate Crowd Counting**: Our custom CBAM-enhanced CNN model precisely counts people in images
- **Crowd Size Classification**: Automatically categorizes crowds as Small, Medium, or Large
- **Stampede Zone Detection**: Identifies high-risk areas with dense, rapidly changing crowds
- **Flow Visualization**: Shows potential crowd movement direction
- **Population Concentration Heatmaps**: Highlights high-density areas in a grid view
- **Adaptive Risk Detection**: Dynamically finds risky zones without fixed thresholds
- **Hotspot Clustering**: Discovers high-density regions using DBSCAN
- **Crowd Fingerprint Profiles**: Analyzes spatial patterns in crowds

## 🏗️ Our Approach

We built a powerful deep learning model with these key innovations:

### Model Architecture
- **VGG16 Backbone**: Uses a pre-trained network for strong feature extraction
- **Dual-Branch Decoder**: Two separate branches with 3×3 kernels (for small heads) and 5×5 kernels (for medium heads)
- **CBAM Attention Modules**: Smart attention mechanisms that help the model focus on important parts of the image
- **Skip Connections**: Preserves spatial information and improves accuracy
- **Patch-Based Training**: Efficiently processes high-resolution images by training on 512×512 patches

### Training Strategy
- Resizes images to 1024×1024
- Extracts random 512×512 patches (acts as data augmentation)
- Uses a custom hybrid loss function (MSE + Focal Loss) for better performance
- Achieves an MAE (Mean Absolute Error) of 21.52 on Shanghai Tech-B dataset

## 🚀 How to Use It

### Option 1: Quick Start (Windows)
1. Double-click `run_app.bat`
2. Open http://localhost:4001 in your browser
3. Upload an image and get your crowd count!

### Option 2: Manual Setup

#### Backend Setup
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

#### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

## 🌐 Try It Live!

Visit the deployed application at: https://crowd-counting.shivaaydhondiyal.online

## 📊 Performance

Our model delivers:
- Near-accurate predictions across video frames
- Excellent performance on medium and large-sized heads
- Temporal stability for video inference

## 💡 Project Novelties

1. **Dual-Kernel Multi-Scale Extraction**: Instead of traditional multi-column designs
2. **CBAM Attention Integration**: Lightweight adaptive attention modules
3. **Patch-Based Training**: Memory-efficient training on high-res images
4. **Skip Connections**: Reduces information loss and overfitting
5. **Custom Hybrid Loss**: Combines MSE and Focal Loss for better accuracy
6. **Adaptive Risk Analysis**: No fixed thresholds, adapts to crowd variations

## 📝 License

MIT
