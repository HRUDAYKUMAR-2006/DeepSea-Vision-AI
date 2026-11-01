# DeepSea Vision AI: Search Party Quick-Start Manual

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
