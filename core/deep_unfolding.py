r"""
Detection-Aware Deep Unfolding Network Module
============================================
Direct implementation of Slides 10, 12, and 13:
1. Optimization Problem:
      J* = argmin_J  0.5 * || I - (J*t + B(1-t)) ||^2 + \lambda R(J) + \mu L_det(J)

2. Unrolled Stage k (repeated K=3 times):
      Z_k = J_{k-1} - \rho_k * \nabla f(J_{k-1})
      J_k = ProxNet_k(Z_k, F_det)

Stage progression per Slide 12:
  - Input: Dull, hazy, color-shifted image.
  - Stage 1: Color cast fixed (lost red is restored using the water model).
  - Stage 2: Haze removed and contrast raised.
  - Stage 3: Fish/marine edges sharpened, guided by detector features F_det.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Optional
from core.image_formation import UnderwaterImageFormation, DegradationEstimator
from core.detector import MarineObjectDetector


class ProxNetStage:
    """
    Learned Proximal Network (ProxNet_k) for stage k.
    Replaces the handcrafted regularizer R(J) and integrates detector guidance F_det:
       J_k = ProxNet_k(Z_k, F_det)
    """

    def __init__(self, stage_index: int, rho: float = 0.3):
        self.stage_index = stage_index  # 1, 2, or 3
        self.rho = rho

    def forward(
        self,
        Z_k: np.ndarray,
        F_det: np.ndarray,
        transmission: np.ndarray,
        background_light: np.ndarray
    ) -> np.ndarray:
        """
        Executes stage-specific proximal projection guided by F_det and underwater physics.
        """
        Z = np.clip(Z_k.copy(), 0.0, 1.0)
        H, W, C = Z.shape

        if self.stage_index == 1:
            # ----------------------------------------------------
            # STAGE 1 (Slide 12): "Color cast fixed: lost red is restored using the water model"
            # ----------------------------------------------------
            # Water model attenuation compensation:
            # Red channel attenuates much faster than green and blue.
            # Compensate red deficit based on green-blue radiance and transmission map t(x)
            mean_gb = (Z[:, :, 1] + Z[:, :, 2]) * 0.5
            red_deficit = np.maximum(0.0, mean_gb - Z[:, :, 0])
            t_ref = np.clip(transmission[:, :, 0], 0.10, 0.95)

            # Physics restoration: deficit is scaled by water absorption (1 - t)
            red_boost = red_deficit * (1.0 - t_ref) * 1.15
            J1 = Z.copy()
            J1[:, :, 0] = np.clip(Z[:, :, 0] + red_boost, 0.0, 1.0)

            # Color constancy & chromatic adjustment
            avg_r = np.mean(J1[:, :, 0]) + 1e-4
            avg_g = np.mean(J1[:, :, 1]) + 1e-4
            avg_b = np.mean(J1[:, :, 2]) + 1e-4
            avg_gray = (avg_r + avg_g + avg_b) / 3.0

            scale_r = np.clip(avg_gray / avg_r, 0.75, 1.35)
            scale_g = np.clip(avg_gray / avg_g, 0.85, 1.15)
            scale_b = np.clip(avg_gray / avg_b, 0.80, 1.20)

            J1[:, :, 0] = np.clip(J1[:, :, 0] * scale_r, 0.0, 1.0)
            J1[:, :, 1] = np.clip(J1[:, :, 1] * scale_g, 0.0, 1.0)
            J1[:, :, 2] = np.clip(J1[:, :, 2] * scale_b, 0.0, 1.0)
            return J1

        elif self.stage_index == 2:
            # ----------------------------------------------------
            # STAGE 2 (Slide 12): "Haze removed and contrast raised"
            # ----------------------------------------------------
            # Invert backscatter veil: J = (Z - B*(1-t)) / max(t, t_0)
            t_safe = np.maximum(transmission, 0.25)
            B = background_light.reshape(1, 1, 3)

            # Physics dehazing
            dehazed_direct = (Z - B * (1.0 - t_safe) * 0.85) / t_safe
            dehazed_direct = np.clip(dehazed_direct, 0.0, 1.0)

            # Blend with Z to maintain stable signal-to-noise ratio
            dehazed = 0.75 * dehazed_direct + 0.25 * Z

            # Dynamic contrast enhancement via CLAHE in LAB space
            lab = cv2.cvtColor((dehazed * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            enhanced_lab = cv2.merge([l_enhanced, a, b])
            J2 = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
            return np.clip(J2, 0.0, 1.0)

        elif self.stage_index == 3:
            # ----------------------------------------------------
            # STAGE 3 (Slide 12): "Fish edges sharpened, because the detector asks for them"
            # ----------------------------------------------------
            # Feature guidance F_det from detector backbone
            if F_det.ndim == 3 and F_det.shape[2] > 0:
                det_edge_guide = np.mean(np.abs(F_det), axis=2)
                d_min, d_max = det_edge_guide.min(), det_edge_guide.max()
                det_edge_guide = (det_edge_guide - d_min) / (d_max - d_min + 1e-6)
            else:
                det_edge_guide = np.zeros((H, W), dtype=np.float32)

            # High-frequency sharpening modulated by detection interest
            blurred = cv2.GaussianBlur(Z, (3, 3), 1.0)
            high_freq = Z - blurred

            guide_3d = np.repeat(det_edge_guide[:, :, np.newaxis], 3, axis=2)
            sharpened = Z + (0.8 * guide_3d + 0.4) * high_freq

            # Edge-preserving bilateral smoothing of background noise
            sharp_uint8 = (np.clip(sharpened, 0.0, 1.0) * 255).astype(np.uint8)
            refined = cv2.bilateralFilter(sharp_uint8, d=5, sigmaColor=35, sigmaSpace=35)
            J3 = refined.astype(np.float32) / 255.0
            return np.clip(J3, 0.0, 1.0)

        else:
            return Z


class DetectionAwareDeepUnfoldingNetwork:
    """
    Complete Detection-Aware Deep Unfolding Network (K stages).
    Orchestrates:
      - Degradation estimator (t, B)
      - Unfolded optimization stages (Physics step + Detection guidance + ProxNet)
      - Intermediate stage tracking (J0 -> J1 -> J2 -> J3)
      - End-to-end integration with detector feedback
    """

    def __init__(
        self,
        num_stages: int = 3,
        step_sizes: Optional[List[float]] = None
    ):
        self.K = num_stages
        self.step_sizes = step_sizes if step_sizes is not None else [0.35, 0.25, 0.15]
        assert len(self.step_sizes) >= self.K, "Must supply step size rho for each stage"

        self.formation = UnderwaterImageFormation()
        self.estimator = DegradationEstimator()
        self.stages = [ProxNetStage(stage_index=k + 1, rho=self.step_sizes[k]) for k in range(self.K)]

    def forward(
        self,
        degraded_img: np.ndarray,
        detector: MarineObjectDetector
    ) -> Dict[str, Any]:
        """
        Executes deep unfolding inference:
        Input: Degraded image I (H, W, 3)
        Returns dictionary containing:
          - 'I': Original degraded input
          - 't': Estimated transmission map
          - 'B': Estimated background light
          - 'stages': List of intermediate outputs [J_1, J_2, J_3]
          - 'enhanced_image': Final output J_K (J_hat)
          - 'F_det': Detector guidance features
        """
        I = np.clip(degraded_img.astype(np.float32), 0.0, 1.0)

        # 1. Degradation Estimator: predict t and B
        t, B = self.estimator.predict(I)

        # 2. Extract initial detector features F_det from degraded input
        F_det = detector.extract_detector_guidance_features(I)

        # 3. Unfold K stages
        J_prev = I.copy()
        stage_outputs = []
        physics_step_outputs = []

        for k in range(self.K):
            rho_k = self.step_sizes[k]
            stage_net = self.stages[k]

            # Physics step: Z_k = J_{k-1} - \rho_k * \nabla f(J_{k-1})
            Z_k = self.formation.physics_step(J_prev, I, t, B, rho=rho_k)
            physics_step_outputs.append(Z_k)

            # If k >= 1, update F_det using the intermediate enhanced features
            if k > 0:
                F_det = detector.extract_detector_guidance_features(J_prev)

            # ProxNet: J_k = ProxNet_k(Z_k, F_det)
            J_k = stage_net.forward(Z_k, F_det, t, B)
            stage_outputs.append(J_k)

            # Prepare for next stage
            J_prev = J_k

        enhanced_image = stage_outputs[-1]

        return {
            "I": I,
            "t": t,
            "B": B,
            "F_det": F_det,
            "physics_steps": physics_step_outputs,
            "stage_outputs": stage_outputs,
            "enhanced_image": enhanced_image
        }


if __name__ == "__main__":
    detector = MarineObjectDetector()
    unfolding_net = DetectionAwareDeepUnfoldingNetwork(num_stages=3)

    # Simulated hazy underwater input
    sample_img = np.zeros((160, 160, 3), dtype=np.float32)
    sample_img[:, :] = [0.1, 0.45, 0.55] # Cyan underwater background
    cv2.circle(sample_img, (80, 80), 30, (0.3, 0.6, 0.65), -1) # Faint fish

    res = unfolding_net.forward(sample_img, detector)
    print("Unfolding completed successfully!")
    print(f"Num stages: {len(res['stage_outputs'])}")
    for i, s in enumerate(res['stage_outputs']):
        print(f"Stage {i+1} Mean: {s.mean():.4f}, Std: {s.std():.4f}")
    print("Enhanced Image Mean:", res['enhanced_image'].mean())
