# DeepSea Vision AI: Detection-Aware Deep Unfolding Platform

[![Live Demo](https://img.shields.io/badge/Live_Website-GitHub_Pages-0284c7?style=for-the-badge&logo=googlechrome&logoColor=white)](https://hrudaykumar-2006.github.io/DeepSea-Vision-AI/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PWA Ready](https://img.shields.io/badge/PWA-Installable_App-10b981?style=for-the-badge&logo=pwa&logoColor=white)](#offline-pwa-app-installation)
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)](LICENSE)

> **Live Web Application:** [https://hrudaykumar-2006.github.io/DeepSea-Vision-AI/](https://hrudaykumar-2006.github.io/DeepSea-Vision-AI/)  
> **Standalone Download Bundle:** [DeepSea_Vision_AI_SearchParty_Edition.zip](dist/DeepSea_Vision_AI_SearchParty_Edition.zip)

---

## 1. How to Enable GitHub Pages (Make the Website Publicly Accessible)

Anyone who visits your repository can use the web application directly in their browser without installing anything!

### Simple 30-Second Setup:
1. Go to your repository settings: [https://github.com/HRUDAYKUMAR-2006/DeepSea-Vision-AI/settings/pages](https://github.com/HRUDAYKUMAR-2006/DeepSea-Vision-AI/settings/pages)
2. Under **"Build and deployment"** $\to$ **"Source"**, select **"Deploy from a branch"**.
3. Under **"Branch"**, select **`main`** and choose folder **`/docs`**, then click **Save**.
4. Within 1 minute, your website will be live at:
   $$\text{\bf https://hrudaykumar-2006.github.io/DeepSea-Vision-AI/}$$
5. (Optional) In the repository homepage, click the ⚙️ gear icon next to **"About"** on the right sidebar, check **"Use your GitHub Pages website"**, and save. Now a direct **"View Website"** link appears prominently at the top of your repository!

---

## 2. System Overview

Underwater optical imagery suffers from severe color distortion, backscatter haze, low contrast, and blur caused by wavelength-dependent light absorption and scattering in water. Traditional vision pipelines apply image enhancement first and object detection second as disjoint steps; consequently, enhancement is tuned solely for human visual appeal rather than machine detection utility, often wiping out subtle high-frequency morphological features required for marine organism localization.

This platform implements the **Detection-Aware Deep Unfolding Network**. It unrolls the physical optimization model of underwater image formation into $K=3$ learnable stages. High-level detector feedback $F_{\text{det}}$ guides each unrolled proximal stage, and the enhancer and detector are optimized jointly end-to-end. Furthermore, the imaging camera operates strictly within an extended physical **deep-sea operational envelope (0m to 500m depth)** and rated **hydrostatic pressure boundaries up to 56 atm**.

---

## 3. Mathematical Formulations (Applied Directly from PPT)

### A. Underwater Optical Degradation Model *(Slide 13, Equation 1)*
The observed degraded underwater image $I(x)$ at spatial coordinate $x$ is given by:
$$I(x) = J(x) \cdot t(x) + B \cdot (1 - t(x))$$
where:
- $I(x)$: Observed degraded underwater image (hazy, color-shifted, low contrast).
- $J(x)$: True clear scene radiance (the target to recover).
- $t(x) \in [0, 1]$: Medium transmission map, characterizing residual light attenuation along the optical path:
  $$t(x) = \exp(-\beta(\lambda) \cdot d(x))$$
  with spectral attenuation coefficients $\beta_R = 0.350\,\text{m}^{-1}, \beta_G = 0.080\,\text{m}^{-1}, \beta_B = 0.050\,\text{m}^{-1}$.
- $B \in [0, 1]^3$: Homogeneous background water light vector (ambient backscatter veil).

### B. Global Optimization Problem *(Slide 13, Equation 2)*
Image reconstruction is formulated as an energy minimization problem that balances physical data fidelity, regularizer prior, and marine detector feedback:
$$J^* = \arg\min_J \frac{1}{2} \big\| I - \big(J \cdot t + B(1 - t)\big) \big\|^2 + \lambda R(J) + \mu L_{\text{det}}(J)$$
where:
- $\frac{1}{2} \| I - (J \cdot t + B(1 - t)) \|^2 = f(J)$: Physics-based data-fidelity term.
- $R(J)$: Regularization prior penalizing unphysical artifacts.
- $L_{\text{det}}(J)$: Detection task loss backpropagated from the detector to guide enhancement.

### C. Deep Unrolled Stage Formulation *(Slide 13, Equation 3)*
Using the proximal gradient method, the optimization objective is unrolled into $K$ iterative stages ($k = 1, \dots, K$ where $K = 3$):
$$Z_k = J_{k-1} - \rho_k \nabla f(J_{k-1})$$
$$J_k = \text{ProxNet}_k(Z_k, F_{\text{det}})$$
The analytical gradient of the physics data-fidelity term is:
$$\nabla f(J_{k-1}) = - \big(I - (J_{k-1} \cdot t + B(1 - t))\big) \cdot t$$
yielding the physics update:
$$Z_k = J_{k-1} + \rho_k \cdot t \cdot \big[ I - (J_{k-1} \cdot t + B(1 - t)) \big]$$
where:
- $\rho_k$: Learnable stage step size.
- $F_{\text{det}}$: Detection feedback feature map extracted from the detector backbone/neck.
- $\text{ProxNet}_k$: Learned proximal convolutional mapping substituting the handcrafted regularizer.

### D. Joint End-to-End Training Loss *(Slide 10)*
The network is jointly optimized using the composite multi-task loss:
$$L = L_{\text{rec}} + \alpha L_{\text{perc}} + \beta L_{\text{det}} + \gamma L_{\text{stage}}$$
where:
- $L_{\text{rec}} = \| \hat{J} - J_{\text{gt}} \|_1$: Direct pixel reconstruction loss between enhanced image $\hat{J}$ and ground truth.
- $L_{\text{perc}} = \| \nabla \hat{J} - \nabla J_{\text{gt}} \|_1$: Perceptual / high-frequency edge fidelity loss.
- $L_{\text{det}}$: Marine object detection task loss (CIoU bounding box localization + class cross-entropy + objectness).
- $L_{\text{stage}} = \sum_{k=1}^K w_k \| J_k - J_{\text{gt}} \|^2$: Multi-stage deep supervision loss across intermediate unrolled iterations.
- Default weighting hyperparameters: $\alpha = 0.2, \beta = 0.5, \gamma = 0.3$.

---

## 4. Extended 500-Meter Depth & Hydrostatic Pressure Constraints

The camera profile is rated for deep-sea operations from the surface down to the mesopelagic zone ($500\,\text{m}$):

### Hydrostatic & Atmospheric Pressure Law
At depth $d$ (meters) below sea level, the total ambient pressure $P(d)$ experienced by the camera housing is:
$$P(d) = P_0 + \frac{\rho \cdot g \cdot d}{101325} \quad (\text{atm})$$
where:
- $P_0 = 1.0\,\text{atm} = 101.325\,\text{kPa}$ (Sea-level atmospheric pressure).
- $\rho = 1025\,\text{kg/m}^3$ (Standard seawater density).
- $g = 9.80665\,\text{m/s}^2$ (Gravitational acceleration).
- Rating envelope: $d \in [0.0\,\text{m}, 500.0\,\text{m}]$, corresponding to $P \in [1.0\,\text{atm}, 50.60\,\text{atm}]$ ($\approx 5127\,\text{kPa}$).
- Titanium housing maximum limit: **$56.0\,\text{atm}$** ($\approx 550\,\text{m}$).
- Safety telemetry monitoring:
  - $0\text{ m} \le d \le 490\text{ m}$ ($P < 50.4\text{ atm}$): `SAFE`
  - $490\text{ m} < d \le 500\text{ m}$ ($50.4\text{ atm} \le P \le 56.0\text{ atm}$): `WARNING_HIGH_PRESSURE`
  - $d > 500\text{ m}$ ($P > 56.0\text{ atm}$): `OUT_OF_RANGE` (Safety shutdown trigger)

---

## 5. Visual Progression Across Unfolded Stages *(Slide 12)*

| Stage | Operation & Principle | Visual Outcome |
|---|---|---|
| **0. Input $I(x)$** | Optical capture at depth $d$ | Dull, hazy, severe color-shift, marine objects obscured |
| **1. Stage 1 ($J_1$)** | Color cast fixed: lost red restored using water model | Red channel restored proportionally to $(1 - t(x))$; chromatic balance restored |
| **2. Stage 2 ($J_2$)** | Haze removed and contrast raised | Backscatter veil $B(1-t)$ inverted; dynamic range expanded via local CLAHE |
| **3. Stage 3 ($J_3$)** | Marine edges sharpened via detector feedback $F_{\text{det}}$ | High-frequency contour sharpening modulated by detector saliency |
| **4. Output** | Marine YOLO Detector + NMS post-processing | Accurate bounding boxes, classes (`fish`, `turtle`, `echinoderm`, `crustacean`), high confidence (e.g. `Fish 0.98`) |

---

## 6. How to Run Locally

### A. Run Automated Unit & Integration Tests
```bash
python -m pytest tests/test_pipeline.py -v
```

### B. Launch the Local Web Dashboard
```bash
python demo/app.py
```
Open **`http://127.0.0.1:5000`** in your browser.

### C. 1-Click Desktop App (Windows & Linux)
- **Windows:** Double-click `run_app.bat`
- **Linux/Mac:** Run `./run_app.sh`
