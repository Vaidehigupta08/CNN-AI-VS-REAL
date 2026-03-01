# 🔍 CNN AI vs Real Image Detector

A full-stack deep learning application that detects whether an image is **AI-generated** or a **real photograph** using a CNN model built with **MobileNetV2** transfer learning.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38-red)

## 📁 Project Structure

```
CNN-AI-VS-REAL/
├── main.py              # FastAPI backend (REST API)
├── app.py               # Streamlit frontend (Web UI)
├── test_api.py          # API test script
├── test_model.py        # Model test script
├── requirements.txt     # Python dependencies
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

> **Note:** The model file (`cnn_ai_real_model_fixed.h5` ~11MB) is not included in the repo. See [Model Setup](#-model-setup) below.

## 🧠 Model Details

| Property | Value |
|----------|-------|
| **Architecture** | MobileNetV2 (Transfer Learning) |
| **Input Size** | 224 × 224 RGB |
| **Output** | Binary (AI vs Real) |
| **Framework** | TensorFlow / Keras |

**Classification Thresholds:**
| Probability | Prediction |
|:-----------:|:----------:|
| < 0.35 | 🤖 AI Generated |
| 0.35 - 0.65 | ❓ Uncertain |
| > 0.65 | 📷 Real Image |

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/Vaidehigupta08/CNN-AI-VS-REAL.git
cd CNN-AI-VS-REAL
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Model Setup
Download/train the model and place `cnn_ai_real_model_fixed.h5` in the project root.

**From Google Colab:**
```python
model.save('cnn_ai_real_model_fixed.h5')
from google.colab import files
files.download('cnn_ai_real_model_fixed.h5')
```

### 4. Start the Backend API
```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```
API runs at: **http://127.0.0.1:8000** | Docs: **http://127.0.0.1:8000/docs**

### 5. Start the Frontend
```bash
streamlit run app.py
```
UI runs at: **http://127.0.0.1:8501**

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Server status |
| GET | `/health` | Health check |
| GET | `/info` | API info |
| POST | `/predict` | Upload image & get prediction |
| GET | `/docs` | Swagger UI documentation |

### Example: Predict via cURL
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@image.jpg"
```

**Response:**
```json
{
  "prediction": "Real Image",
  "confidence": 0.853,
  "probability": 0.8528
}
```

## 🖥️ Frontend Features

- Drag & drop image upload
- Live API status indicator
- Color-coded results (🟢 Real, 🔴 AI, 🟡 Uncertain)
- Confidence percentage & probability bar
- Responsive dark-themed UI

## 🛠️ Tech Stack

- **Backend:** FastAPI + Uvicorn
- **Frontend:** Streamlit
- **ML Model:** TensorFlow + Keras (MobileNetV2)
- **Image Processing:** Pillow + NumPy

## 📝 License

This project is for educational purposes.

## 👤 Author

**Vaidehi Gupta** — [@Vaidehigupta08](https://github.com/Vaidehigupta08)
