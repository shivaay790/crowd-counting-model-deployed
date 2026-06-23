# Crowd Counting Application

A modern web application for counting people in images using computer vision.

## Project Structure

```
4_crowd_counting_deployed/
├── backend/          # Flask backend with ONNX model inference
├── frontend/         # React + TypeScript frontend
├── .github/          # GitHub workflows for deployment
├── run_app.bat       # Script to run both backend and frontend locally
└── README.md
```

## Tech Stack

### Backend
- **Framework**: Flask
- **Inference**: ONNX Runtime
- **Storage**: AWS S3 (for model weights)
- **Fallback API**: Hugging Face Gradio

### Frontend
- **Framework**: React 19
- **Language**: TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS
- **Animations**: Framer Motion
- **Icons**: Lucide React

## Getting Started

### Prerequisites
- Python 3.8+
- Node.js 18+
- npm or yarn

### Running Locally

#### Option 1: Using the provided script (Windows)
```bash
run_app.bat
```

#### Option 2: Manual setup

**Backend:**
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

## Environment Variables

### Backend (backend/.env)
```
PORT=4000
DEBUG=false
ALLOWED_ORIGINS=https://crowd-counting.shivaaydhondiyal.online,http://localhost:4001,http://localhost:5173
BUCKET_NAME=crowd-counting-model-weights
AWS_REGION=eu-north-1
MODEL_S3_KEY=model.onnx
MODEL_URL=
```

### Frontend (frontend/.env)
```
VITE_API_URL=http://localhost:4000
```

## API Endpoints

- `GET /` - Health check
- `POST /count` - Count people from uploaded image file
- `POST /count-base64` - Count people from base64 encoded image

## License

MIT
