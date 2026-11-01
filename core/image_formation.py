r"""
Image Formation Model & Degradation Estimator
==============================================
Direct implementation of Slide 10 & Slide 13 formulas:
1. Underwater Optical Degradation Model:
      I(x) = J(x) * t(x) + B * (1 - t(x))
   where:
      I(x) = Observed degraded underwater image
      J(x) = Clear scene radiance (to recover)
      t(x) = Medium transmission map in [0, 1]
      B    = Homogeneous background light vector (ambient backscatter)

2. Degradation Estimator:
   Predicts transmission map t(x) and background light B from degraded input I.

3. Physics Data-Fidelity Gradient Step:
      f(J) = 0.5 * || I - (J * t + B * (1 - t)) ||^2
      \nabla f(J) = - (I - (J * t + B * (1 - t))) * t
      Z_k = J_{k-1} - \rho_k * \nabla f(J_{k-1})
          = J_{k-1} + \rho_k * t * (I - (J_{k-1} * t + B * (1 - t)))
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import cv2
from typing import Tuple, Dict, Any, Optional


class UnderwaterImageFormation:
    """
    Manages the physical underwater image formation model,
    forward degradation synthesis, and data-fidelity gradient computations.
    """

    def __init__(self, t_min: float = 0.05, t_max: float = 0.98):
        self.t_min = t_min
        self.t_max = t_max

    def forward_degradation(
        self,
        clean_image: np.ndarray,
        transmission: np.ndarray,
        background_light: np.ndarray
    ) -> np.ndarray:
        """
        Applies: I(x) = J(x) * t(x) + B * (1 - t(x))
        clean_image: float32 in [0, 1], shape (H, W, C)
        transmission: float32 in [0, 1], shape (H, W, 1) or (H, W, C)
        background_light: float32 in [0, 1], shape (C,) or (1, 1, C)
        Returns: degraded image I in [0, 1]
        """
        J = np.clip(clean_image.astype(np.float32), 0.0, 1.0)
        t = np.clip(transmission.astype(np.float32), self.t_min, self.t_max)
        B = np.clip(background_light.astype(np.float32).reshape(1, 1, -1), 0.0, 1.0)

        degraded = J * t + B * (1.0 - t)
        return np.clip(degraded, 0.0, 1.0)

    def compute_data_fidelity_gradient(
        self,
        J: np.ndarray,
        I: np.ndarray,
        t: np.ndarray,
        B: np.ndarray
    ) -> np.ndarray:
        """
        Computes \nabla f(J) = - (I - (J * t + B * (1 - t))) * t
        """
        reconstructed = J * t + B * (1.0 - t)
        residual = I - reconstructed
        grad = - (residual * t)
        return grad

    def physics_step(
        self,
        J_prev: np.ndarray,
        I: np.ndarray,
        t: np.ndarray,
        B: np.ndarray,
        rho: float
    ) -> np.ndarray:
        """
        Executes one gradient descent step on the data-fidelity term:
        Z_k = J_{k-1} - \rho_k * \nabla f(J_{k-1})
        """
        grad = self.compute_data_fidelity_gradient(J_prev, I, t, B)
        Z_k = J_prev - (rho * grad)
        return np.clip(Z_k, 0.0, 1.0)


class DegradationEstimator:
    """
    Sub-network / Estimator module (Slide 10: Degradation estimator).
    Predicts:
      - Transmission map t(x) [per channel or spatial]
      - Background light B [R, G, B]
    from input degraded image I.
    """

    def __init__(self, omega: float = 0.95, patch_size: int = 15):
        self.omega = omega
        self.patch_size = patch_size

    def estimate_background_light(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Estimates the homogeneous background water light B.
        In underwater environments, the background light is dominated by the backscatter,
        brightest in the green-blue channels in open water regions.
        """
        img = np.clip(img_rgb.astype(np.float32), 0.0, 1.0)
        H, W, C = img.shape
        num_pixels = H * W
        num_candidates = max(int(num_pixels * 0.001), 10)

        # In underwater imagery, dark channel or red-attenuation channel guides candidate selection
        # Red channel is heavily absorbed; pixels where red is low and G/B are high represent water backscatter
        water_backscatter_metric = (img[:, :, 1] + img[:, :, 2]) - img[:, :, 0]
        flat_metric = water_backscatter_metric.ravel()
        candidate_indices = np.argpartition(flat_metric, -num_candidates)[-num_candidates:]

        flat_img = img.reshape(-1, C)
        candidates = flat_img[candidate_indices]
        # Robust estimate: median of candidate backscatter pixels
        B = np.median(candidates, axis=0)
        
        # Ensure physical bounds
        B = np.clip(B, 0.05, 0.98)
        return B

    def estimate_transmission(
        self,
        img_rgb: np.ndarray,
        background_light: np.ndarray,
        guided_filter_radius: int = 20,
        eps: float = 1e-3
    ) -> np.ndarray:
        """
        Estimates the wavelength-dependent transmission map t(x).
        Uses red-channel underwater transmission prior refined with edge-preserving guided filter.
        """
        img = np.clip(img_rgb.astype(np.float32), 0.0, 1.0)
        H, W, C = img.shape

        # Normalize by background light
        B_reshaped = background_light.reshape(1, 1, 3) + 1e-6
        normalized = img / B_reshaped

        # Underwater prior: Transmission is primarily determined by maximum of green/blue versus red
        # Red channel attenuates fastest: 1 - min_patch(1 - normalized_red)
        # Using minimum filter across localized patch
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.patch_size, self.patch_size))
        
        # Dark channel of attenuation
        min_channel = np.min(normalized, axis=2)
        dark_patch = cv2.erode(min_channel, kernel)
        raw_transmission = 1.0 - (self.omega * dark_patch)
        raw_transmission = np.clip(raw_transmission, 0.05, 0.98)

        # Refine transmission using grayscale guide to preserve structural edges
        gray_guide = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        refined_t = cv2.ximgproc.guidedFilter(
            guide=gray_guide,
            src=raw_transmission.astype(np.float32),
            radius=guided_filter_radius,
            eps=eps
        ) if hasattr(cv2, 'ximgproc') else cv2.bilateralFilter(raw_transmission.astype(np.float32), 9, 75, 75)

        refined_t = np.clip(refined_t, 0.05, 0.98)
        # Return transmission with shape (H, W, 1) or channel-specific transmission
        t_channel = np.repeat(refined_t[:, :, np.newaxis], 3, axis=2)
        return t_channel

    def predict(self, degraded_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        End-to-end degradation estimation: returns (transmission_map, background_light)
        """
        B = self.estimate_background_light(degraded_img)
        t = self.estimate_transmission(degraded_img, B)
        return t, B


if __name__ == "__main__":
    # Quick test of image formation and degradation estimator
    sample_clean = np.ones((128, 128, 3), dtype=np.float32) * 0.7
    # Draw a colored patch (simulating fish)
    sample_clean[40:80, 40:80] = [0.9, 0.4, 0.1] # Orange fish

    formation = UnderwaterImageFormation()
    synth_t = np.ones((128, 128, 3), dtype=np.float32) * 0.6
    synth_B = np.array([0.1, 0.5, 0.7], dtype=np.float32)

    degraded = formation.forward_degradation(sample_clean, synth_t, synth_B)
    estimator = DegradationEstimator()
    est_t, est_B = estimator.predict(degraded)

    print("Synthesized Degraded Mean:", degraded.mean())
    print("Estimated Background Light B:", est_B)
    print("Estimated Transmission Mean:", est_t.mean())

    # Gradient step check
    grad = formation.compute_data_fidelity_gradient(degraded, degraded, est_t, est_B)
    print("Gradient norm at initial state:", np.linalg.norm(grad))
    step_result = formation.physics_step(degraded, degraded, est_t, est_B, rho=0.3)
    print("Physics step completed successfully. Shape:", step_result.shape)
