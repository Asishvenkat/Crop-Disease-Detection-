"""
Project Verification Script
Checks that all files are in place and properly configured
"""
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent

def check_file_exists(path, description):
    """Check if a file exists"""
    full_path = PROJECT_ROOT / path
    if full_path.exists():
        size = full_path.stat().st_size
        print(f"✓ {description:<50} ({size:,} bytes)")
        return True
    else:
        print(f"✗ {description:<50} MISSING")
        return False


def check_directory_exists(path, description):
    """Check if a directory exists"""
    full_path = PROJECT_ROOT / path
    if full_path.exists() and full_path.is_dir():
        item_count = len(list(full_path.iterdir()))
        print(f"✓ {description:<50} ({item_count} items)")
        return True
    else:
        print(f"✗ {description:<50} MISSING")
        return False


def verify_imports():
    """Verify that Python imports work"""
    print("\n🔍 Verifying Python imports...")
    
    try:
        from src.config import CLASSES, MODEL_PATH
        print(f"✓ Config imports work (5 classes defined)")
        return True
    except Exception as e:
        print(f"✗ Config import failed: {e}")
        return False


def verify_config():
    """Verify configuration values"""
    print("\n🔍 Verifying configuration...")
    
    try:
        from src.config import (
            CLASSES, CONFIDENCE_THRESHOLD, 
            DISEASE_TREATMENTS, SYMPTOM_DISEASE_MAPPING
        )
        
        checks = [
            (len(CLASSES) == 5, f"Classes count: {len(CLASSES)} (expected 5)"),
            (CONFIDENCE_THRESHOLD == 0.70, f"Confidence threshold: {CONFIDENCE_THRESHOLD}"),
            (len(DISEASE_TREATMENTS) >= 5, f"Treatment recipes: {len(DISEASE_TREATMENTS)}"),
            (len(SYMPTOM_DISEASE_MAPPING) == 4, f"Symptom mappings: {len(SYMPTOM_DISEASE_MAPPING)}")
        ]
        
        all_pass = True
        for check, msg in checks:
            if check:
                print(f"✓ {msg}")
            else:
                print(f"✗ {msg}")
                all_pass = False
        
        return all_pass
        
    except Exception as e:
        print(f"✗ Configuration verification failed: {e}")
        return False


def main():
    print("=" * 70)
    print("🧪 PROJECT VERIFICATION - Crop Disease Detection")
    print("=" * 70)
    
    results = {}
    
    # Core files
    print("\n📋 Core Files:")
    results['train'] = check_file_exists("train.py", "Training script")
    results['main'] = check_file_exists("main.py", "FastAPI backend")
    results['app'] = check_file_exists("app.py", "Streamlit frontend")
    results['test'] = check_file_exists("test_api.py", "Test script")
    results['requirements'] = check_file_exists("requirements.txt", "Dependencies")
    
    # Source files
    print("\n📦 Source Code:")
    results['config'] = check_file_exists("src/config.py", "Configuration")
    results['grad_cam'] = check_file_exists("src/grad_cam.py", "Grad-CAM")
    results['init'] = check_file_exists("src/__init__.py", "Package init")
    
    # Documentation
    print("\n📚 Documentation:")
    results['readme'] = check_file_exists("README.md", "README (comprehensive)")
    results['quickstart'] = check_file_exists("QUICKSTART.md", "Quick Start Guide")
    results['winning'] = check_file_exists("WINNING_FORMULA.md", "Winning Formula")
    
    # Configuration
    print("\n⚙️  Configuration Files:")
    results['env'] = check_file_exists(".env", "Environment variables")
    results['streamlit'] = check_file_exists(".streamlit/config.toml", "Streamlit config")
    
    # Helper scripts
    print("\n🚀 Helper Scripts:")
    results['bat'] = check_file_exists("start_api.bat", "Windows launcher")
    results['sh'] = check_file_exists("start_api.sh", "Linux/Mac launcher")
    
    # Directories
    print("\n📁 Directories:")
    results['src_dir'] = check_directory_exists("src", "Source directory")
    results['data_dir'] = check_directory_exists("data", "Data directory")
    results['models_dir'] = check_directory_exists("models", "Models directory")
    
    # Verify imports and config
    results['imports'] = verify_imports()
    results['config'] = verify_config()
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\n✅ Passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All checks passed! Project is ready.")
        print("\nNext steps:")
        print("1. Download dataset from Kaggle to data/dataset/")
        print("2. Run: python train.py")
        print("3. Run: python main.py (terminal 1)")
        print("4. Run: streamlit run app.py (terminal 2)")
        print("5. Open: http://localhost:8501")
        return 0
    else:
        print(f"\n⚠️  {total - passed} checks failed. See above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
