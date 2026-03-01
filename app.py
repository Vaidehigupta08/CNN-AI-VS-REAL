import streamlit as st
import requests
from PIL import Image
import os

# -----------------------------
# Config
# -----------------------------
API_URL = os.environ.get("API_URL", "https://guptavaidehi-cnn-ai-real-api.hf.space")

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(
    page_title="AI vs Real Image Detector",
    page_icon="🔍",
    layout="centered"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .result-card {
        background: linear-gradient(135deg, #1e1e2f, #2d2d44);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0;
    }
    .prediction-label {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    .ai-label { color: #ff4b4b; }
    .real-label { color: #00c853; }
    .uncertain-label { color: #ffa726; }
    .confidence-text {
        font-size: 1.2rem;
        color: #ccc;
    }
    .info-box {
        background: #1a1a2e;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
    }
    .footer {
        text-align: center;
        color: #666;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown('<p class="main-title">🔍 AI vs Real Image Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload an image to detect whether it is AI-generated or a real photograph</p>', unsafe_allow_html=True)

# -----------------------------
# Sidebar - API Status
# -----------------------------
with st.sidebar:
    st.header("⚙️ API Status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        if health.get("model_loaded"):
            st.success("✅ API Online | Model Loaded")
        else:
            st.warning("⚠️ API Online | Model NOT Loaded")
    except:
        st.error("❌ API Offline")

    st.divider()
    st.header("ℹ️ About")
    st.markdown("""
    This app uses a **CNN model** built with **MobileNetV2** 
    to classify images as:
    - 🤖 **AI Generated**
    - 📷 **Real Image**
    - ❓ **Uncertain**
    
    **Model:** MobileNetV2 (Transfer Learning)  
    **Input Size:** 224 × 224  
    **Backend:** FastAPI + TensorFlow
    """)

    st.divider()
    st.header("📊 Confidence Guide")
    st.markdown("""
    | Probability | Prediction |
    |:-----------:|:----------:|
    | < 0.35 | 🤖 AI Generated |
    | 0.35 - 0.65 | ❓ Uncertain |
    | > 0.65 | 📷 Real Image |
    """)

# -----------------------------
# Image Upload
# -----------------------------
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Supported formats: JPG, JPEG, PNG, WEBP"
    )

# -----------------------------
# Display & Predict
# -----------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Show image centered
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption=f"📁 {uploaded_file.name}")

    # Predict button centered
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_btn = st.button("🔍 Analyze Image", use_container_width=True, type="primary")

    if predict_btn:
        with st.spinner("🧠 Analyzing image..."):
            try:
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type or "image/png"
                    )
                }
                response = requests.post(f"{API_URL}/predict", files=files, timeout=30)

                if response.status_code == 200:
                    data = response.json()
                    prediction = data["prediction"]
                    confidence = data["confidence"]
                    probability = data["probability"]

                    # Choose color and emoji
                    if prediction == "AI Generated":
                        color_class = "ai-label"
                        emoji = "🤖"
                        bar_color = "#ff4b4b"
                    elif prediction == "Real Image":
                        color_class = "real-label"
                        emoji = "📷"
                        bar_color = "#00c853"
                    else:
                        color_class = "uncertain-label"
                        emoji = "❓"
                        bar_color = "#ffa726"

                    # Result card
                    st.markdown(f"""
                    <div class="result-card">
                        <div style="font-size: 3rem;">{emoji}</div>
                        <div class="prediction-label {color_class}">{prediction}</div>
                        <div class="confidence-text">Confidence: {confidence * 100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Progress bar
                    st.markdown("**Probability Distribution**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("🤖 AI Generated", f"{(1 - probability) * 100:.1f}%")
                    with col2:
                        st.metric("📷 Real Image", f"{probability * 100:.1f}%")

                    st.progress(probability)

                elif response.status_code == 503:
                    st.error("🚫 Model not loaded. Please check the backend.")
                else:
                    st.error(f"❌ Error: {response.json().get('detail', 'Unknown error')}")

            except requests.exceptions.ConnectionError:
                st.error("🔌 Cannot connect to API. Make sure the backend is running on port 8000.")
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. The server might be busy.")
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")

else:
    # Placeholder when no image uploaded
    st.markdown("""
    <div style="text-align: center; padding: 3rem; border: 2px dashed #444; border-radius: 16px; margin: 2rem 0;">
        <div style="font-size: 3rem;">📤</div>
        <p style="color: #888; font-size: 1.1rem;">Drag and drop an image above or click to browse</p>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<div class="footer">
    Built with ❤️ using Streamlit + FastAPI + TensorFlow<br>
    AI vs Real Image Detection | CNN Project
</div>
""", unsafe_allow_html=True)