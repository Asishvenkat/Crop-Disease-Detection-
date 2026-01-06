"""
Quick test script to verify the setup
Run this after training to test backend prediction
"""
import requests
import json
from pathlib import Path
import time

API_URL = "http://localhost:8000"
TEST_IMAGE_PATH = Path("test_image.jpg")  # Update with your test image

def test_health():
    """Test API health endpoint"""
    print("\n🔍 Testing API health...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        print(f"✓ Status: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_classes():
    """Test classes endpoint"""
    print("\n🔍 Testing classes endpoint...")
    try:
        response = requests.get(f"{API_URL}/classes", timeout=5)
        print(f"✓ Status: {response.status_code}")
        classes = response.json()
        print(f"✓ Classes: {classes['count']}")
        for cls in classes['classes']:
            print(f"  - {cls}")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_info():
    """Test info endpoint"""
    print("\n🔍 Testing info endpoint...")
    try:
        response = requests.get(f"{API_URL}/info", timeout=5)
        print(f"✓ Status: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_predict():
    """Test prediction endpoint with sample image"""
    print("\n🔍 Testing prediction endpoint...")
    
    # Check if test image exists
    if not TEST_IMAGE_PATH.exists():
        print(f"⚠️  Test image not found: {TEST_IMAGE_PATH}")
        print("   To test predictions, provide a test image or use the Streamlit UI")
        return False
    
    try:
        with open(TEST_IMAGE_PATH, 'rb') as f:
            files = {'image': f}
            data = {
                'yellowing_leaves': True,
                'brown_spots': False,
                'wilting': False,
                'white_fungal_growth': False
            }
            
            print(f"  Sending image: {TEST_IMAGE_PATH}")
            response = requests.post(f"{API_URL}/predict", files=files, data=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Status: {response.status_code}")
            print(f"  Disease: {result['disease']}")
            print(f"  Confidence: {result['confidence_percent']}")
            print(f"  Severity: {result['severity_level']} ({result['severity_score']:.0f}/100)")
            print(f"  Message: {result['message']}")
            return True
        else:
            print(f"✗ Status: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 70)
    print("🧪 CROP DISEASE DETECTION - API TEST SUITE")
    print("=" * 70)
    print("\nℹ️  Make sure the API is running: python main.py")
    print("=" * 70)
    
    time.sleep(1)
    
    tests = [
        ("Health Check", test_health),
        ("Classes Endpoint", test_classes),
        ("Info Endpoint", test_info),
        ("Prediction Endpoint", test_predict)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except KeyboardInterrupt:
            print("\n⛔ Test interrupted by user")
            break
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            results[test_name] = False
        
        time.sleep(0.5)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:7} | {test_name}")
    
    print("=" * 70)
    print(f"Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! API is ready for use.")
    else:
        print("❌ Some tests failed. Check API and dataset setup.")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
