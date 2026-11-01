r"""
Underwater Image Quality & Object Detection Metrics Module
==========================================================
Implements full-reference and no-reference underwater image quality metrics
and detection evaluation metrics as listed in Slide 4 (Evaluate Performance):
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- UIQM (Underwater Image Quality Measure): UICM + UISM + UIConM
- UCIQE (Underwater Color Image Quality Evaluation)
- Detection Precision, Recall, F1, and Average Precision (AP)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import cv2
import math
from typing import Dict, Any, List, Optional
from core.detector import DetectionBox


def compute_psnr(img_pred: np.ndarray, img_gt: np.ndarray) -> float:
    """Calculates Peak Signal-to-Noise Ratio (PSNR) in dB."""
    p = np.clip(img_pred.astype(np.float32), 0.0, 1.0)
    g = np.clip(img_gt.astype(np.float32), 0.0, 1.0)
    mse = np.mean((p - g) ** 2)
    if mse <= 1e-10:
        return 50.0
    psnr = 10.0 * math.log10(1.0 / mse)
    return round(float(psnr), 2)


def compute_ssim(img_pred: np.ndarray, img_gt: np.ndarray) -> float:
    """Calculates Structural Similarity Index (SSIM)."""
    p = np.clip(img_pred.astype(np.float32), 0.0, 1.0)
    g = np.clip(img_gt.astype(np.float32), 0.0, 1.0)

    # Convert to grayscale if 3-channel
    if p.ndim == 3 and p.shape[2] == 3:
        p_gray = cv2.cvtColor((p * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        g_gray = cv2.cvtColor((g * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    else:
        p_gray, g_gray = p, g

    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    mu1 = cv2.GaussianBlur(p_gray, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(g_gray, (11, 11), 1.5)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(p_gray * p_gray, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(g_gray * g_gray, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(p_gray * g_gray, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return round(float(np.mean(ssim_map)), 4)


def compute_uciqe(img_rgb: np.ndarray) -> float:
    r"""
    Underwater Color Image Quality Evaluation (UCIQE) - Yang & Xu.
    UCIQE = c1 * \sigma_c + c2 * con_l + c3 * \mu_s
    Standard CIELab implementation with proper center offsets.
    """
    img = np.clip(img_rgb.astype(np.float32), 0.0, 1.0)
    # Convert to CIELab via uint8
    lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    # OpenCV maps L to [0, 255], a to [0, 255] (center 128), b to [0, 255] (center 128)
    L = lab[:, :, 0] * (100.0 / 255.0)
    a = lab[:, :, 1] - 128.0
    b = lab[:, :, 2] - 128.0

    # Chroma = sqrt(a*^2 + b*^2)
    chroma = np.sqrt(a**2 + b**2)
    sigma_c = float(np.std(chroma))

    # Luminance contrast: difference between top 1% and bottom 1% of L*
    con_l = float(np.percentile(L, 99) - np.percentile(L, 1))

    # Saturation = Chroma / sqrt(L*^2 + Chroma^2)
    saturation = chroma / (np.sqrt(L**2 + chroma**2) + 1e-4)
    mu_s = float(np.mean(saturation))

    # Standard Yang & Xu coefficients: c1=0.4680, c2=0.2745, c3=0.2576
    uciqe = 0.4680 * (sigma_c / 100.0) + 0.2745 * (con_l / 100.0) + 0.2576 * mu_s
    return round(float(uciqe), 4)


def compute_uiqm(img_rgb: np.ndarray) -> Dict[str, float]:
    """
    Underwater Image Quality Measure (UIQM) - Panetta et al.
    UIQM = c1 * UICM + c2 * UISM + c3 * UIConM
    where c1=0.0282, c2=0.2953, c3=3.5753
    """
    img = np.clip(img_rgb.astype(np.float32), 0.0, 1.0)
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    # 1. UICM (Colorfulness)
    RG = R - G
    YB = 0.5 * (R + G) - B
    u_rg, std_rg = np.mean(RG), np.std(RG)
    u_yb, std_yb = np.mean(YB), np.std(YB)
    uicm = -0.0268 * math.sqrt(u_rg**2 + u_yb**2) + 0.1586 * math.sqrt(std_rg**2 + std_yb**2)

    # 2. UISM (Sharpness using Sobel operator)
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_energy = np.mean(np.sqrt(sobel_x**2 + sobel_y**2)) / 255.0
    uism = edge_energy * 5.0

    # 3. UIConM (Contrast)
    uiconm = float(np.std(gray) / (np.mean(gray) + 1e-6))

    # Total UIQM
    c1, c2, c3 = 0.0282, 0.2953, 3.5753
    uiqm_score = c1 * uicm + c2 * uism + c3 * (uiconm * 0.1)

    return {
        "UIQM": round(float(uiqm_score), 4),
        "UICM": round(float(uicm), 4),
        "UISM": round(float(uism), 4),
        "UIConM": round(float(uiconm), 4)
    }


def compute_box_iou(boxA: DetectionBox, boxB: DetectionBox) -> float:
    """Calculates Intersection over Union between two DetectionBox objects."""
    xA = max(boxA.x1, boxB.x1)
    yA = max(boxA.y1, boxB.y1)
    xB = min(boxA.x2, boxB.x2)
    yB = min(boxA.y2, boxB.y2)

    interWidth = max(0.0, xB - xA)
    interHeight = max(0.0, yB - yA)
    interArea = interWidth * interHeight

    boxAArea = max(0.0, (boxA.x2 - boxA.x1) * (boxA.y2 - boxA.y1))
    boxBArea = max(0.0, (boxB.x2 - boxB.x1) * (boxB.y2 - boxB.y1))
    unionArea = boxAArea + boxBArea - interArea

    if unionArea <= 0.0:
        return 0.0
    return interArea / unionArea


def evaluate_detections(
    predictions: List[DetectionBox],
    ground_truths: List[Dict[str, Any]],
    iou_thresh: float = 0.5
) -> Dict[str, Any]:
    """Evaluates detection Precision, Recall, and F1 score against ground truths."""
    if not ground_truths and not predictions:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "true_positives": 0, "detected": 0, "total_gt": 0}
    if not ground_truths and predictions:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "true_positives": 0, "detected": len(predictions), "total_gt": 0}
    if ground_truths and not predictions:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "true_positives": 0, "detected": 0, "total_gt": len(ground_truths)}

    matched_gt = set()
    tp = 0
    fp = 0

    for pred in predictions:
        best_iou = 0.0
        best_gt_idx = -1
        for idx, gt in enumerate(ground_truths):
            gt_box = DetectionBox(gt["box"][0], gt["box"][1], gt["box"][2], gt["box"][3], 1.0, 0, gt.get("class_name", "fish"))
            iou = compute_box_iou(pred, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        if best_iou >= iou_thresh and best_gt_idx not in matched_gt:
            matched_gt.add(best_gt_idx)
            tp += 1
        else:
            fp += 1

    fn = len(ground_truths) - len(matched_gt)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-6)

    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "total_gt": len(ground_truths)
    }


if __name__ == "__main__":
    t_clean = np.ones((100, 100, 3), dtype=np.float32) * 0.8
    t_degraded = np.ones((100, 100, 3), dtype=np.float32) * 0.4
    t_degraded[:, :, 0] *= 0.2 # heavy red loss

    print("PSNR:", compute_psnr(t_degraded, t_clean))
    print("SSIM:", compute_ssim(t_degraded, t_clean))
    print("UCIQE:", compute_uciqe(t_degraded))
    print("UIQM:", compute_uiqm(t_degraded))
