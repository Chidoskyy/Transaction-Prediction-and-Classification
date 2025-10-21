#!/usr/bin/env python3
"""
Bank Statement ML Models UI Launcher
Simple script to launch the Streamlit UI
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_ui.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def launch_ui():
    """Launch the Streamlit UI"""
    print("🚀 Launching Bank Statement ML Models UI...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "model_ui.py", "--server.port", "8501"])
    except KeyboardInterrupt:
        print("\n👋 UI closed by user")
    except Exception as e:
        print(f"❌ Error launching UI: {e}")

def main():
    """Main launcher function"""
    print("🏦 Bank Statement ML Models UI Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("model_ui.py"):
        print("❌ Error: model_ui.py not found in current directory")
        print("Please run this script from the same directory as model_ui.py")
        return
    
    # Install requirements
    if not install_requirements():
        return
    
    # Launch UI
    launch_ui()

if __name__ == "__main__":
    main()
