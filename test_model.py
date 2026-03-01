from tensorflow.keras.models import load_model

print("Testing model load...")

try:
    model = load_model("C:\\Users\\vaide\\OneDrive\\Documents\\cnn project 2\\cnn_ai_real_model_new.keras")
    print("[SUCCESS] Model loaded successfully!")
    print(f"Model summary:")
    model.summary()
except Exception as e:
    print(f"[ERROR] {str(e)}")
    import traceback
    traceback.print_exc()
