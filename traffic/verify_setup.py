#!/usr/bin/env python3
"""
iSENSOR Environment Verification & Setup Helper
Comprehensive check of all components and PyTorch compatibility
"""

import sys
import torch
from pathlib import Path

def print_header(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def check_pytorch():
    """Check PyTorch version and compatibility"""
    print_header("PyTorch Version Check")
    
    print(f"PyTorch Version: {torch.__version__}")
    
    # Check if 2.6+
    version_parts = torch.__version__.split('.')
    major, minor = int(version_parts[0]), int(version_parts[1].split('+')[0])
    
    if major > 2 or (major == 2 and minor >= 6):
        print("Status: PyTorch 2.6+ (uses safe_globals)")
        print("safe_globals available:", hasattr(torch.serialization, 'safe_globals'))
    elif major >= 2:
        print(f"Status: PyTorch {major}.{minor} (uses weights_only=False)")
    else:
        print("Status: PyTorch < 2.0 (old version, may have issues)")
    
    return True

def check_packages():
    """Check all required packages"""
    print_header("Package Dependencies Check")
    
    packages = {
        'torch': 'PyTorch',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'yaml': 'YAML',
        'PIL': 'Pillow',
        'streamlit': 'Streamlit',
        'pandas': 'Pandas',
    }
    
    missing = []
    for pkg_name, display_name in packages.items():
        try:
            mod = __import__(pkg_name)
            if hasattr(mod, '__version__'):
                version = mod.__version__
                print(f"✓ {display_name:20} {version}")
            else:
                print(f"✓ {display_name:20} (installed)")
        except ImportError:
            print(f"✗ {display_name:20} MISSING")
            missing.append(pkg_name)
    
    return len(missing) == 0, missing

def check_models():
    """Check available model weights"""
    print_header("Model Weights Check")
    
    models = {
        'yolov7-tiny.pt': 'YOLOv7-Tiny (12 MB - Fast)',
        'yolov7.pt': 'YOLOv7 (37 MB - Balanced)',
        'yolov7x.pt': 'YOLOv7-X (71 MB - Accurate)',
    }
    
    available = []
    for model_file, description in models.items():
        path = Path(model_file)
        if path.exists():
            size_mb = path.stat().st_size / (1024*1024)
            print(f"✓ {description:40} ({size_mb:6.1f} MB)")
            available.append(model_file)
        else:
            print(f"✗ {description:40} NOT FOUND")
    
    return available, len(available) > 0

def check_model_loading():
    """Test actual model loading"""
    print_header("Model Loading Test")
    
    available_models, _ = check_models()
    if not available_models:
        print("⊘ No models available - skipping load test")
        return False
    
    try:
        print(f"Testing load of: {available_models[0]}")
        from models.experimental import attempt_load
        
        device = torch.device('cpu')
        model = attempt_load(available_models[0], map_location=device)
        
        params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model loaded successfully!")
        print(f"  Parameters: {params/1e6:.1f}M")
        print(f"  Device: {device}")
        print(f"  Status: Ready for inference")
        return True
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False

def check_files():
    """Check if key files exist"""
    print_header("Project Files Check")
    
    files = {
        'detect.py': 'Main detection script',
        'detect_or_track.py': 'Detection + tracking script',
        'streamlit_app.py': 'Streamlit web interface',
        'download_and_test.py': 'Model download script',
        'models/yolo.py': 'YOLOv7 model definition',
        'utils/general.py': 'Utilities (NMS, etc)',
        'sample.mp4': 'Sample video file',
    }
    
    all_exist = True
    for filename, description in files.items():
        path = Path(filename)
        if path.exists():
            if path.is_file():
                size = path.stat().st_size
                if size < 1024:
                    size_str = f"{size}B"
                elif size < 1024*1024:
                    size_str = f"{size/1024:.1f}KB"
                else:
                    size_str = f"{size/(1024*1024):.1f}MB"
                print(f"✓ {description:40} ({size_str})")
            else:
                print(f"✓ {description:40} (directory)")
        else:
            print(f"✗ {description:40} NOT FOUND")
            all_exist = False
    
    return all_exist

def generate_report():
    """Generate comprehensive status report"""
    print_header("iSENSOR Environment Verification Report")
    
    results = {}
    
    # Run all checks
    results['pytorch'] = check_pytorch()
    results['packages'], missing_pkgs = check_packages()
    results['models'], has_models = check_models()
    results['load_test'] = check_model_loading()
    results['files'] = check_files()
    
    # Summary
    print_header("Summary & Recommendations")
    
    if all(results.values()):
        print("✓ ALL SYSTEMS GO! Ready to use iSENSOR")
        print()
        print("Next steps:")
        print("  1. Try detection: python detect.py --weights yolov7-tiny.pt --source sample.mp4")
        print("  2. Run Streamlit: streamlit run streamlit_app.py")
        print("  3. Check help: python detect.py --help")
    else:
        print("⚠️  Some issues detected:")
        print()
        
        if not results['pytorch']:
            print("  • PyTorch issue - may need reinstall")
        
        if not results['packages']:
            print(f"  • Missing packages: {', '.join(missing_pkgs)}")
            print("    Run: pip install -r requirements.txt")
        
        if not results['models']:
            print("  • No model weights found")
            print("    Run: python download_and_test.py")
        
        if not results['load_test']:
            print("  • Model loading failed")
            print("    Check PyTorch version compatibility")
        
        if not results['files']:
            print("  • Some project files missing")
            print("    Check repository is complete")
    
    print()
    print("="*70)
    print(f"Report generated: PyTorch {torch.__version__}")
    print("="*70)

if __name__ == "__main__":
    try:
        generate_report()
    except Exception as e:
        print(f"\n❌ Error during verification: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
