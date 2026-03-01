from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import os
import json
import zipfile

# -----------------------------
# 1. App create
# -----------------------------
app = FastAPI(
    title="AI vs Real Image API",
    description="Detects AI-generated images vs real images",
    version="1.0.0"
)

# -----------------------------
# 2. Load model (only once)
# -----------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cnn_ai_real_model_fixed.h5")
IMG_SIZE = (224, 224)

model = None

# Auto-fix model if batch_shape issue exists (only for .keras files)
def fix_model_if_needed(model_path):
    """Auto-fix batch_shape issues in keras format files"""
    # Skip if not .keras file
    if not model_path.endswith('.keras'):
        return False

    if not os.path.exists(model_path):
        return False

    try:
        with zipfile.ZipFile(model_path, 'r') as z:
            if 'config.json' not in z.namelist():
                return False

            config_str = z.read('config.json').decode('utf-8')
            metadata_str = z.read('metadata.json').decode('utf-8')
            weights_data = z.read('model.weights.h5')

            config = json.loads(config_str)

        # Check if batch_shape exists
        has_batch_shape = False
        def check_batch_shape(obj):
            nonlocal has_batch_shape
            if isinstance(obj, dict):
                if 'batch_shape' in obj:
                    has_batch_shape = True
                for value in obj.values():
                    check_batch_shape(value)
            elif isinstance(obj, list):
                for item in obj:
                    check_batch_shape(item)

        check_batch_shape(config)

        if not has_batch_shape:
            return False

        # Fix batch_shape and dtype issues
        def clean_config(obj):
            if isinstance(obj, dict):
                if 'batch_shape' in obj:
                    obj.pop('batch_shape')
                if 'dtype' in obj and isinstance(obj['dtype'], dict):
                    if 'config' in obj['dtype'] and 'name' in obj['dtype']['config']:
                        obj['dtype'] = obj['dtype']['config']['name']
                for value in obj.values():
                    clean_config(value)
            elif isinstance(obj, list):
                for item in obj:
                    clean_config(item)

        clean_config(config)

        # Create fixed model
        fixed_path = model_path.replace('.keras', '_fixed.keras')
        with zipfile.ZipFile(fixed_path, 'w', zipfile.ZIP_DEFLATED) as z_new:
            z_new.writestr('config.json', json.dumps(config))
            z_new.writestr('metadata.json', metadata_str)
            z_new.writestr('model.weights.h5', weights_data)

        # Replace original with fixed
        os.remove(model_path)
        os.rename(fixed_path, model_path)
        print(f"[AUTO-FIX] Fixed batch_shape and dtype issues in model")
        return True
    except Exception as e:
        print(f"[AUTO-FIX] Error: {e}")
        return False

# Try to fix model before loading
fix_model_if_needed(MODEL_PATH)

try:
    model = load_model(MODEL_PATH)
    print("[SUCCESS] Model loaded successfully!")
except Exception as e:
    print(f"[WARNING] Model loading failed: {e}")
    print("Please save model from Colab: model.save('/content/cnn_ai_real_model_final.h5')")
    model = None

# -----------------------------
# 3. Helper: image preprocess
# -----------------------------
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# -----------------------------
# 4. Health & Info endpoints
# -----------------------------
@app.get("/")
async def root():
    return {
        "status": "running",
        "model_loaded": model is not None,
        "endpoints": ["/docs", "/predict", "/health", "/info"]
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.get("/info")
async def info():
    return {
        "title": "AI vs Real Image API",
        "version": "1.0.0",
        "image_size": IMG_SIZE,
        "model_path": MODEL_PATH,
        "model_loaded": model is not None
    }

# -----------------------------
# 5. Prediction endpoint
# -----------------------------
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please save the model from Colab.")

    try:
        # read image bytes
        image_bytes = await file.read()

        # preprocess
        img = preprocess_image(image_bytes)

        # model prediction
        prob = float(model.predict(img, verbose=0)[0][0])

        # threshold logic
        if prob < 0.35:
            prediction = "AI Generated"
        elif prob > 0.65:
            prediction = "Real Image"
        else:
            prediction = "Uncertain"

        # response
        return {
            "prediction": prediction,
            "confidence": round(prob, 3),
            "probability": prob
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")