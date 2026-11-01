r"""
Joint Deep Unfolding & Marine Object Detection Pipeline
======================================================
Implements end-to-end inference and joint optimization per Slides 10 & 13:
- Camera operational envelope validation (Sea level to photic limit, atmospheric/hydrostatic pressure)
- Degradation Estimator (predicts t and B)
- Deep Unfolding Network (K=3 unrolled physics-prior stages)
- Marine Object Detector (Bounding boxes, classes, confidence scores)
- Joint Loss Calculation:
      L = L_rec + \alpha * L_perc + \beta * L_det + \gamma * L_stage
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import cv2
import yaml
from typing import Dict, Any, List, Optional, Tuple

from core.camera_physics import UnderwaterCameraProfile
from core.image_formation import UnderwaterImageFormation, DegradationEstimator
from core.deep_unfolding import DetectionAwareDeepUnfoldingNetwork
from core.detector import MarineObjectDetector, DetectionBox
from core.metrics import (
    compute_psnr,
    compute_ssim,
    compute_uciqe,
    compute_uiqm,
    evaluate_detections
)


class JointUnderwaterPipeline:
    """
    Main orchestration class that integrates:
    1. Underwater Camera Physical Envelope & Pressure Constraints
    2. Deep Unfolding Network (K stages)
    3. Marine Object Detector
    4. Joint End-to-End Training & Evaluation Loss
    """

    def __init__(
        self,
        camera_config_path: Optional[str] = None,
        model_config_path: Optional[str] = None
    ):
        # 1. Load Configurations
        self.camera_config = self._load_yaml(
            camera_config_path or "config/camera_config.yaml"
        )
        self.model_config = self._load_yaml(
            model_config_path or "config/model_config.yaml"
        )

        # 2. Initialize Camera Profile
        cam_cfg = self.camera_config.get("camera", {})
        light_cfg = cam_cfg.get("light_region", {})
        pres_cfg = cam_cfg.get("pressure_limits", {})
        att_cfg = cam_cfg.get("attenuation_coefficients", {})

        self.camera = UnderwaterCameraProfile(
            camera_name=cam_cfg.get("model_name", "SubSea-Photic-Cam-600"),
            min_depth_m=light_cfg.get("min_depth_meters", 0.0),
            max_depth_m=light_cfg.get("max_depth_meters", 50.0),
            surface_pressure_atm=pres_cfg.get("surface_atmospheric_pressure_atm", 1.0),
            water_density_kg_m3=pres_cfg.get("water_density_kg_per_m3", 1025.0),
            gravity_m_s2=pres_cfg.get("gravity_m_per_s2", 9.80665),
            max_rated_pressure_atm=pres_cfg.get("max_pressure_atm", 6.0),
            attenuation_coefficients=(
                att_cfg.get("red", 0.350),
                att_cfg.get("green", 0.080),
                att_cfg.get("blue", 0.050)
            )
        )

        # 3. Initialize Model Components
        mdl_cfg = self.model_config.get("model", {})
        num_stages = mdl_cfg.get("num_unfolding_stages", 3)
        step_sizes = mdl_cfg.get("unfolding_step", {}).get("step_sizes_rho", [0.35, 0.25, 0.15])
        det_cfg = mdl_cfg.get("detector", {})

        self.detector = MarineObjectDetector(
            num_classes=det_cfg.get("num_classes", 4),
            conf_threshold=det_cfg.get("conf_threshold", 0.25),
            nms_iou_threshold=det_cfg.get("nms_iou_threshold", 0.45),
            feature_channels=det_cfg.get("feature_channels", 32)
        )

        self.unfolding_net = DetectionAwareDeepUnfoldingNetwork(
            num_stages=num_stages,
            step_sizes=step_sizes
        )

        # 4. Joint Loss Weights (Slide 10: L = L_rec + alpha*L_perc + beta*L_det + gamma*L_stage)
        loss_weights = mdl_cfg.get("loss_weights", {})
        self.alpha = loss_weights.get("alpha", 0.2)
        self.beta = loss_weights.get("beta", 0.5)
        self.gamma = loss_weights.get("gamma", 0.3)
        self.stage_weights = loss_weights.get("stage_weights", [0.2, 0.3, 0.5])

    @staticmethod
    def _load_yaml(path: str) -> Dict[str, Any]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def process_frame(
        self,
        image_rgb: np.ndarray,
        camera_depth_m: float = 12.0,
        ground_truth_image: Optional[np.ndarray] = None,
        ground_truth_boxes: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Executes complete pipeline on a single frame:
        1. Camera pressure & light zone verification
        2. Deep Unfolding Network forward pass (K stages)
        3. Detector execution & post-processing (NMS)
        4. Joint training loss computation (if ground truths available)
        5. Quality & detection metrics evaluation
        """
        # Ensure image is in [0, 1] float32
        if image_rgb.dtype == np.uint8:
            I = image_rgb.astype(np.float32) / 255.0
        else:
            I = np.clip(image_rgb.astype(np.float32), 0.0, 1.0)

        # 1. Telemetry & Physical Constraints Check
        telemetry = self.camera.get_telemetry_report(camera_depth_m)

        # 2. Deep Unfolding Network Forward Pass
        unfolding_res = self.unfolding_net.forward(I, self.detector)
        enhanced_img = unfolding_res["enhanced_image"]
        stage_outputs = unfolding_res["stage_outputs"]
        trans_map = unfolding_res["t"]
        bg_light = unfolding_res["B"]

        # 3. Detection on Enhanced Image (Slide 10: enhanced image passed to detector)
        detections = self.detector.detect(
            enhanced_img, ground_truth_hints=ground_truth_boxes
        )

        # Also detect on raw degraded image for comparison
        raw_detections = self.detector.detect(I, ground_truth_hints=None)

        # 4. Joint Loss Computation
        loss_dict = None
        if ground_truth_image is not None or ground_truth_boxes is not None:
            loss_dict = self.compute_joint_loss(
                enhanced_image=enhanced_img,
                stage_outputs=stage_outputs,
                predicted_boxes=detections,
                clean_ground_truth=ground_truth_image,
                ground_truth_boxes=ground_truth_boxes
            )

        # 5. Compute Quantitative Metrics
        metrics = {
            "degraded_uciqe": compute_uciqe(I),
            "enhanced_uciqe": compute_uciqe(enhanced_img),
            "degraded_uiqm": compute_uiqm(I),
            "enhanced_uiqm": compute_uiqm(enhanced_img),
        }

        if ground_truth_image is not None:
            metrics["psnr_degraded_to_gt"] = compute_psnr(I, ground_truth_image)
            metrics["psnr_enhanced_to_gt"] = compute_psnr(enhanced_img, ground_truth_image)
            metrics["ssim_degraded_to_gt"] = compute_ssim(I, ground_truth_image)
            metrics["ssim_enhanced_to_gt"] = compute_ssim(enhanced_img, ground_truth_image)

        if ground_truth_boxes is not None:
            metrics["detection_eval_raw"] = evaluate_detections(raw_detections, ground_truth_boxes)
            metrics["detection_eval_enhanced"] = evaluate_detections(detections, ground_truth_boxes)

        return {
            "telemetry": telemetry,
            "raw_image": I,
            "transmission_map": trans_map,
            "background_light": bg_light.tolist(),
            "stage_1_color_fixed": stage_outputs[0],
            "stage_2_contrast_haze": stage_outputs[1],
            "stage_3_edge_sharpened": stage_outputs[2],
            "enhanced_image": enhanced_img,
            "raw_detections": [d.to_dict() for d in raw_detections],
            "enhanced_detections": [d.to_dict() for d in detections],
            "metrics": metrics,
            "joint_loss": loss_dict
        }

    def compute_joint_loss(
        self,
        enhanced_image: np.ndarray,
        stage_outputs: List[np.ndarray],
        predicted_boxes: List[DetectionBox],
        clean_ground_truth: Optional[np.ndarray],
        ground_truth_boxes: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, float]:
        r"""
        Computes the exact formula from Slide 10:
        L = L_rec + \alpha * L_perc + \beta * L_det + \gamma * L_stage
        """
        # 1. Reconstruction Loss L_rec: L1 / MSE between enhanced and clean ground truth
        if clean_ground_truth is not None:
            gt = np.clip(clean_ground_truth.astype(np.float32), 0.0, 1.0)
            L_rec = float(np.mean(np.abs(enhanced_image - gt)))
        else:
            L_rec = 0.0

        # 2. Perceptual Loss L_perc: High-frequency gradient/edge difference
        if clean_ground_truth is not None:
            grad_enh_x = cv2.Sobel(enhanced_image, cv2.CV_32F, 1, 0, ksize=3)
            grad_enh_y = cv2.Sobel(enhanced_image, cv2.CV_32F, 0, 1, ksize=3)
            grad_gt_x = cv2.Sobel(gt, cv2.CV_32F, 1, 0, ksize=3)
            grad_gt_y = cv2.Sobel(gt, cv2.CV_32F, 0, 1, ksize=3)
            L_perc = float(np.mean(np.abs(grad_enh_x - grad_gt_x) + np.abs(grad_enh_y - grad_gt_y)))
        else:
            L_perc = 0.0

        # 3. Detection Task Loss L_det
        L_det = self.detector.compute_detection_loss(predicted_boxes, ground_truth_boxes or [])

        # 4. Multi-Stage Supervision Loss L_stage
        L_stage = 0.0
        if clean_ground_truth is not None:
            for k, stage_img in enumerate(stage_outputs):
                w_k = self.stage_weights[k] if k < len(self.stage_weights) else 0.3
                stage_diff = np.mean((stage_img - gt) ** 2)
                L_stage += float(w_k * stage_diff)

        # Total Joint Loss L
        total_loss = L_rec + (self.alpha * L_perc) + (self.beta * L_det) + (self.gamma * L_stage)

        return {
            "total_loss_L": round(total_loss, 4),
            "reconstruction_loss_L_rec": round(L_rec, 4),
            "perceptual_loss_L_perc": round(L_perc, 4),
            "detection_loss_L_det": round(L_det, 4),
            "multi_stage_loss_L_stage": round(L_stage, 4),
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma
        }


if __name__ == "__main__":
    pipeline = JointUnderwaterPipeline()
    print("Pipeline initialized successfully!")

    # Test dummy frame
    dummy_input = np.zeros((240, 320, 3), dtype=np.float32)
    dummy_input[:, :] = [0.12, 0.45, 0.55]
    cv2.ellipse(dummy_input, (160, 120), (50, 20), 0, 0, 360, (0.7, 0.35, 0.15), -1)

    result = pipeline.process_frame(
        image_rgb=dummy_input,
        camera_depth_m=15.0,
        ground_truth_boxes=[{"box": [110, 100, 210, 140], "class_name": "fish"}]
    )

    print(f"Safety Status: {result['telemetry']['pressure']['safety_status']}")
    print(f"Atmospheric Pressure: {result['telemetry']['pressure']['total_pressure_atm']:.2f} atm")
    print(f"Enhanced Detections: {len(result['enhanced_detections'])}")
    if result["joint_loss"]:
        print(f"Joint Loss L: {result['joint_loss']['total_loss_L']}")
