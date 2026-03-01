#!/usr/bin/env python3
"""
SimpleTest script for AI vs Real Image Detection API
"""

import requests
import sys
from pathlib import Path

API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("\n=== Testing Health Endpoint ===")
    try:
        response = requests.get(f"{API_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_info():
    """Test info endpoint"""
    print("\n=== Testing Info Endpoint ===")
    try:
        response = requests.get(f"{API_URL}/info")
        print(f"Status: {response.status_code}")
        import json
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_predict(image_path):
    """Test prediction endpoint"""
    print(f"\n=== Testing Prediction Endpoint ===")
    print(f"Image: {image_path}")

    if not Path(image_path).exists():
        print(f"ERROR: File not found: {image_path}")
        return False

    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{API_URL}/predict", files=files)

        print(f"Status: {response.status_code}")
        import json
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("API Test Suite")
    print("=" * 50)

    # Test health
    health_ok = test_health()

    # Test info
    info_ok = test_info()

    # Test prediction if image provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        pred_ok = test_predict(image_path)
    else:
        print("\n=== Skipping Prediction Test ===")
        print("Usage: python test_api.py <image_path>")
        pred_ok = True

    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    print(f"Health Check: {'PASS' if health_ok else 'FAIL'}")
    print(f"Info Endpoint: {'PASS' if info_ok else 'FAIL'}")
    print(f"Prediction: {'PASS' if pred_ok else 'SKIP'}")

    if health_ok and info_ok:
        print("\n[OK] API is running correctly!")
    else:
        print("\n[FAIL] Some tests failed. Check the server logs.")

