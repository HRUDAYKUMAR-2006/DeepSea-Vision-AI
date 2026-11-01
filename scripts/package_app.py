r"""
Distribution & Standalone App Bundler for Search Party Organizations
====================================================================
Packages the complete platform into an offline, self-contained zip distribution
with 1-click launchers for field laptops and marine vessel control stations.
"""

import os
import sys
import zipfile
import shutil

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DIST_DIR = os.path.join(BASE_DIR, "dist")
os.makedirs(DIST_DIR, exist_ok=True)

ZIP_NAME = "DeepSea_Vision_AI_SearchParty_Edition.zip"
ZIP_PATH = os.path.join(DIST_DIR, ZIP_NAME)

# 1. Create run_app.bat (Windows Launcher)
BAT_CONTENT = """@echo off
title DeepSea Vision AI - Marine Search & Rescue Platform
echo ===============================================================================
echo     DEEPSEA VISION AI - MARINE SEARCH PARTY & UNDERWATER SURVEY SYSTEM
echo ===============================================================================
echo Starting local web engine and mission dashboard...
cd /d "%~dp0"

REM Verify Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in system PATH.
    echo Please install Python 3.10+ from python.org to run this platform.
    pause
    exit /b
)

echo Initializing database storage and starting application server...
start "" "http://127.0.0.1:5000"
python demo/app.py
pause
"""

# 2. Create run_app.sh (Linux/Mac Launcher)
SH_CONTENT = """#!/bin/bash
echo "==============================================================================="
echo "    DEEPSEA VISION AI - MARINE SEARCH PARTY & UNDERWATER SURVEY SYSTEM"
echo "==============================================================================="
cd "$(dirname "$0")"

if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 could not be found. Please install Python 3.10+."
    exit 1
fi

echo "Launching application server on http://127.0.0.1:5000..."
python3 demo/app.py &
sleep 2

if which xdg-open > /dev/null; then
    xdg-open "http://127.0.0.1:5000"
elif which open > /dev/null; then
    open "http://127.0.0.1:5000"
fi
wait
"""

# 3. Create requirements.txt
REQ_CONTENT = """flask>=3.0.0
numpy>=1.24.0
opencv-python>=4.8.0
pyyaml>=6.0
"""

# 4. Create SEARCH_PARTY_MANUAL.md
MANUAL_CONTENT = """# DeepSea Vision AI: Search Party Quick-Start Manual

## Standalone Deployment Instructions for Marine Vessels & Search Teams

### 1. Launching on Windows
Double-click `run_app.bat`. This automatically:
- Checks the local Python runtime.
- Starts the embedded DeepSea Vision AI server.
- Opens your default web browser to `http://127.0.0.1:5000`.

### 2. Launching on Linux / Mac / Marine Terminals
Run in terminal:
```bash
chmod +x run_app.sh
./run_app.sh
```

### 3. Progressive Web App (PWA) Offline Installation
1. Open `http://127.0.0.1:5000` in Google Chrome, Microsoft Edge, or Chromium.
2. Click the **"Install App"** button in the header (or the install icon in your browser URL bar).
3. The platform will install as a standalone desktop application window with full offline access.

### 4. Operational Envelope & Field Telemetry
- **Operational Depths:** 0 meters to 500 meters (Euphotic down to Mesopelagic zone).
- **Rated Pressure Housing:** 56.0 atm (~5,674 kPa).
- **Dual Databases:**
  - `storage/received_data.db`: Stores raw input feeds, GPS coordinates, and search metadata.
  - `storage/enhanced_results.db`: Stores enhanced images, unrolled stages, and detected target catalogs.

### 5. Automated AI Reports
Click **"Generate AI Report"** on any processed frame to produce a structured inspection report ready for transmission to mission command.
"""


def package():
    # Write launcher files
    bat_path = os.path.join(BASE_DIR, "run_app.bat")
    sh_path = os.path.join(BASE_DIR, "run_app.sh")
    req_path = os.path.join(BASE_DIR, "requirements.txt")
    manual_path = os.path.join(BASE_DIR, "SEARCH_PARTY_MANUAL.md")

    with open(bat_path, "w", encoding="utf-8") as f:
        f.write(BAT_CONTENT)
    with open(sh_path, "w", encoding="utf-8") as f:
        f.write(SH_CONTENT)
    with open(req_path, "w", encoding="utf-8") as f:
        f.write(REQ_CONTENT)
    with open(manual_path, "w", encoding="utf-8") as f:
        f.write(MANUAL_CONTENT)

    print("Building distribution zip...")
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)

    files_to_bundle = [
        "run_app.bat",
        "run_app.sh",
        "requirements.txt",
        "SEARCH_PARTY_MANUAL.md",
        "README.md",
        "run_pipeline.py",
        "config/camera_config.yaml",
        "config/model_config.yaml",
        "core/__init__.py",
        "core/camera_physics.py",
        "core/image_formation.py",
        "core/detector.py",
        "core/deep_unfolding.py",
        "core/joint_pipeline.py",
        "core/metrics.py",
        "core/database_manager.py",
        "core/ai_report_generator.py",
        "models_torch/torch_modules.py",
        "demo/app.py",
        "demo/templates/index.html",
        "demo/static/manifest.json",
        "demo/static/sw.js",
        "demo/static/icons/icon-192.png",
        "demo/static/icons/icon-512.png",
        "datasets/generate_dataset.py",
        "datasets/sample_images/dataset_manifest.json"
    ]

    # Create empty core/__init__.py if missing
    core_init = os.path.join(BASE_DIR, "core", "__init__.py")
    if not os.path.exists(core_init):
        with open(core_init, "w") as f:
            f.write("# DeepSea Vision AI Core Package\n")

    with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for rel_file in files_to_bundle:
            full_path = os.path.join(BASE_DIR, rel_file)
            if os.path.exists(full_path):
                zipf.write(full_path, arcname=os.path.join("DeepSea_Vision_AI", rel_file))
            else:
                print(f"Warning: File not found: {rel_file}")

        # Also add sample images if present
        sample_dir = os.path.join(BASE_DIR, "datasets", "sample_images")
        if os.path.exists(sample_dir):
            for fname in os.listdir(sample_dir):
                if fname.endswith(".png"):
                    zipf.write(
                        os.path.join(sample_dir, fname),
                        arcname=os.path.join("DeepSea_Vision_AI", "datasets", "sample_images", fname)
                    )

    size_mb = os.path.getsize(ZIP_PATH) / (1024 * 1024)
    print(f"Standalone application bundled successfully into:")
    print(f" -> {ZIP_PATH} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    package()

