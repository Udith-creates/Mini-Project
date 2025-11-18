#!/usr/bin/env python3
"""
iSENSOR Streamlit Setup & Launch Script
Handles environment setup, dependency installation, and app launch
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a shell command with error handling"""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}\n")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        print(f"✓ {description} - SUCCESS\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} - FAILED")
        print(f"Error: {e}\n")
        return False

def check_requirements():
    """Check if required packages are installed"""
    print("\n" + "="*60)
    print("📦 Checking Python Environment")
    print("="*60)
    
    required_packages = {
        'torch': 'PyTorch',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'streamlit': 'Streamlit',
        'PIL': 'Pillow'
    }
    
    missing = []
    for pkg, name in required_packages.items():
        try:
            __import__(pkg)
            print(f"✓ {name:20} - installed")
        except ImportError:
            print(f"✗ {name:20} - MISSING")
            missing.append(pkg)
    
    return missing

def main():
    print("\n" + "="*60)
    print("🚀 iSENSOR Streamlit Environment Setup")
    print("="*60)
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("\n✗ Python 3.7+ required")
        sys.exit(1)
    
    print(f"\n✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Check requirements
    missing = check_requirements()
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        response = input("\nInstall missing packages? (y/n): ").strip().lower()
        if response == 'y':
            run_command("pip install -r requirements.txt", 
                       "Install base requirements")
            run_command("pip install -r requirements_streamlit.txt", 
                       "Install Streamlit requirements")
    
    # Check model weights
    print("\n" + "="*60)
    print("📥 Checking Model Weights")
    print("="*60)
    
    models = {
        'yolov7-tiny.pt': 'YOLOv7-Tiny (12 MB - Fast)',
        'yolov7.pt': 'YOLOv7 (37 MB - Balanced)',
        'yolov7x.pt': 'YOLOv7-X (71 MB - Accurate)'
    }
    
    available = []
    missing_models = []
    
    for model_file, description in models.items():
        if Path(model_file).exists():
            size = Path(model_file).stat().st_size / (1024*1024)
            print(f"✓ {description:35} ({size:.1f} MB)")
            available.append(model_file)
        else:
            print(f"✗ {description:35} - NOT FOUND")
            missing_models.append(model_file)
    
    if not available:
        print("\n⚠️  No model weights found!")
        response = input("Download models? (y/n): ").strip().lower()
        if response == 'y':
            if Path('download_and_test.py').exists():
                run_command(f"{sys.executable} download_and_test.py", 
                           "Download and test models")
            else:
                print("✗ download_and_test.py not found")
    
    # Check sample videos
    print("\n" + "="*60)
    print("🎬 Checking Sample Videos")
    print("="*60)
    
    sample_videos = list(Path('.').glob('*.mp4'))
    if sample_videos:
        for video in sample_videos[:5]:
            size = video.stat().st_size / (1024*1024)
            print(f"✓ {video.name:40} ({size:.1f} MB)")
    else:
        print("No sample videos found (optional)")
    
    # Summary and launch
    print("\n" + "="*60)
    print("✓ Environment Check Complete!")
    print("="*60)
    
    response = input("\nLaunch Streamlit app now? (y/n): ").strip().lower()
    if response == 'y':
        print("\n🌐 Launching Streamlit app...")
        print("App will open at: http://localhost:8501\n")
        print("Press Ctrl+C to stop the server\n")
        
        try:
            os.system(f"{sys.executable} -m streamlit run streamlit_app.py")
        except KeyboardInterrupt:
            print("\n\n✓ Streamlit app stopped")
    else:
        print("\n📝 To launch manually, run:")
        print("   streamlit run streamlit_app.py")
        print("\n📖 For detailed setup guide, see: STREAMLIT_GUIDE.md")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✓ Setup cancelled")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
