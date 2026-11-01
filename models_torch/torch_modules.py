r"""
PyTorch Deep Unfolding Network & Marine Object Detection Modules
==============================================================
Provides standard PyTorch nn.Module implementations of:
- DegradationEstimatorNet: Predicts t(x) and background light B
- ProxNet: Learned proximal operator J_k = ProxNet_k(Z_k, F_det)
- MarineYOLO: Detection backbone, neck, and multi-scale detection head
- DetectionAwareDeepUnfoldingNet: Unrolls K optimization stages
- JointLoss: L = L_rec + \alpha * L_perc + \beta * L_det + \gamma * L_stage
"""

import math
from typing import List, Dict, Any, Tuple, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class ConvBlock(nn.Module):
        def __init__(self, in_c: int, out_c: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
            super().__init__()
            self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False)
            self.bn = nn.BatchNorm2d(out_c)
            self.act = nn.LeakyReLU(0.1, inplace=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.act(self.bn(self.conv(x)))


    class DegradationEstimatorNet(nn.Module):
        """
        Lightweight sub-network predicting transmission map t and background light B (Slide 10).
        """
        def __init__(self, in_channels: int = 3):
            super().__init__()
            self.features = nn.Sequential(
                ConvBlock(in_channels, 32),
                ConvBlock(32, 64),
                ConvBlock(64, 64),
            )
            # Transmission head (per-pixel transmission map in (0, 1))
            self.trans_head = nn.Sequential(
                ConvBlock(64, 32),
                nn.Conv2d(32, 3, kernel_size=3, padding=1),
                nn.Sigmoid()
            )
            # Background light head (global pooling to 3 RGB scalar values)
            self.bg_head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, 32),
                nn.LeakyReLU(0.1),
                nn.Linear(32, 3),
                nn.Sigmoid()
            )

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            feat = self.features(x)
            t = self.trans_head(feat)
            # Clamp t to prevent numerical singularities: t in [0.05, 0.98]
            t = 0.05 + 0.93 * t
            B = self.bg_head(feat).view(-1, 3, 1, 1)
            return t, B


    class ResBlock(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.conv1 = ConvBlock(channels, channels)
            self.conv2 = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels)
            )
            self.act = nn.LeakyReLU(0.1, inplace=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.act(x + self.conv2(self.conv1(x)))


    class ProxNet(nn.Module):
        """
        Learned Proximal Network (Slide 13: J_k = ProxNet_k(Z_k, F_det)).
        Replaces the handcrafted regularizer R(J) by parameterizing the proximal step.
        Fuses intermediate gradient update Z_k with detector guidance features F_det.
        """
        def __init__(self, in_channels: int = 3, det_channels: int = 32, mid_channels: int = 64):
            super().__init__()
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(in_channels + det_channels, mid_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(0.1, inplace=True)
            )
            self.res_blocks = nn.Sequential(
                ResBlock(mid_channels),
                ResBlock(mid_channels),
                ResBlock(mid_channels)
            )
            self.out_conv = nn.Sequential(
                nn.Conv2d(mid_channels, in_channels, kernel_size=3, padding=1),
                nn.Sigmoid()
            )

        def forward(self, Z_k: torch.Tensor, F_det: torch.Tensor) -> torch.Tensor:
            # Resize F_det if spatial dimensions differ
            if F_det.shape[2:] != Z_k.shape[2:]:
                F_det = F.interpolate(F_det, size=Z_k.shape[2:], mode="bilinear", align_corners=False)
            x_cat = torch.cat([Z_k, F_det], dim=1)
            feat = self.fusion_conv(x_cat)
            feat = self.res_blocks(feat)
            J_k = self.out_conv(feat)
            return J_k


    class MarineDetectorGuidanceNet(nn.Module):
        """
        Marine Object Detection Backbone with feature extraction for F_det.
        """
        def __init__(self, num_classes: int = 4, out_det_channels: int = 32):
            super().__init__()
            self.num_classes = num_classes
            self.layer1 = ConvBlock(3, 32, stride=2)   # 1/2
            self.layer2 = ConvBlock(32, 64, stride=2)  # 1/4
            self.layer3 = ConvBlock(64, 128, stride=2) # 1/8
            self.layer4 = ConvBlock(128, 256, stride=2)# 1/16

            # Feature projection to F_det guidance channels
            self.guidance_proj = nn.Sequential(
                nn.Conv2d(128, out_det_channels, kernel_size=1),
                nn.BatchNorm2d(out_det_channels),
                nn.LeakyReLU(0.1)
            )

            # Detection Head (Box [x, y, w, h] + Objectness [1] + Class Probs [num_classes])
            self.head = nn.Conv2d(256, 3 * (5 + num_classes), kernel_size=1)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            c1 = self.layer1(x)
            c2 = self.layer2(c1)
            c3 = self.layer3(c2)
            c4 = self.layer4(c3)

            # F_det feature representation
            F_det = self.guidance_proj(c3)
            # Upsample F_det to input resolution for proximal fusion
            F_det = F.interpolate(F_det, size=x.shape[2:], mode="bilinear", align_corners=False)

            preds = self.head(c4)
            return preds, F_det


    class PyTorchDeepUnfoldingNetwork(nn.Module):
        """
        End-to-End Deep Unfolding Network in PyTorch.
        Unrolls K stages of:
            f(J) = 0.5 * || I - (J*t + B(1-t)) ||^2
            \nabla f(J_{k-1}) = - (I - (J_{k-1}*t + B(1-t))) * t
            Z_k = J_{k-1} - \rho_k * \nabla f(J_{k-1})
            J_k = ProxNet_k(Z_k, F_det)
        """
        def __init__(self, num_stages: int = 3, num_classes: int = 4):
            super().__init__()
            self.K = num_stages
            self.estimator = DegradationEstimatorNet(in_channels=3)
            self.detector = MarineDetectorGuidanceNet(num_classes=num_classes)

            # Learnable step sizes \rho_k for each stage
            self.rhos = nn.Parameter(torch.tensor([0.35, 0.25, 0.15], dtype=torch.float32))

            # K stage proximal networks
            self.prox_stages = nn.ModuleList([
                ProxNet(in_channels=3, det_channels=32, mid_channels=48)
                for _ in range(num_stages)
            ])

        def forward(self, I: torch.Tensor) -> Dict[str, Any]:
            # 1. Estimate degradation parameters: t and B
            t, B = self.estimator(I)

            # 2. Extract initial detector features
            _, F_det = self.detector(I)

            # 3. Unroll K stages
            J_prev = I
            stage_outputs = []

            for k in range(self.K):
                rho_k = torch.clamp(self.rhos[k], 0.01, 1.0)
                # Physics step:
                reconstructed = J_prev * t + B * (1.0 - t)
                grad_f = - (I - reconstructed) * t
                Z_k = torch.clamp(J_prev - rho_k * grad_f, 0.0, 1.0)

                # ProxNet step with detector guidance:
                J_k = self.prox_stages[k](Z_k, F_det)
                stage_outputs.append(J_k)

                # Feed updated features for next stage
                if k < self.K - 1:
                    _, F_det = self.detector(J_k)
                J_prev = J_k

            enhanced_image = stage_outputs[-1]
            det_preds, _ = self.detector(enhanced_image)

            return {
                "enhanced_image": enhanced_image,
                "stage_outputs": stage_outputs,
                "t": t,
                "B": B,
                "det_preds": det_preds
            }


    class JointLoss(nn.Module):
        r"""
        Slide 10: Joint end-to-end training:
        L = L_rec + \alpha * L_perc + \beta * L_det + \gamma * L_stage
        """
        def __init__(self, alpha: float = 0.2, beta: float = 0.5, gamma: float = 0.3):
            super().__init__()
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma
            self.l1 = nn.L1Loss()
            self.mse = nn.MSELoss()

        def forward(
            self,
            pred_enhanced: torch.Tensor,
            stage_outputs: List[torch.Tensor],
            det_loss: torch.Tensor,
            gt_image: torch.Tensor
        ) -> Dict[str, torch.Tensor]:
            L_rec = self.l1(pred_enhanced, gt_image)

            # High-frequency edge perceptual loss
            sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=pred_enhanced.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
            sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=pred_enhanced.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)

            e_gx = F.conv2d(pred_enhanced, sobel_kernel_x, padding=1, groups=3)
            e_gy = F.conv2d(pred_enhanced, sobel_kernel_y, padding=1, groups=3)
            g_gx = F.conv2d(gt_image, sobel_kernel_x, padding=1, groups=3)
            g_gy = F.conv2d(gt_image, sobel_kernel_y, padding=1, groups=3)
            L_perc = self.l1(e_gx, g_gx) + self.l1(e_gy, g_gy)

            # Stage loss: supervision on intermediate unrolled stages
            L_stage = sum(self.mse(s, gt_image) for s in stage_outputs) / len(stage_outputs)

            total_loss = L_rec + self.alpha * L_perc + self.beta * det_loss + self.gamma * L_stage

            return {
                "total_loss": total_loss,
                "L_rec": L_rec,
                "L_perc": L_perc,
                "L_det": det_loss,
                "L_stage": L_stage
            }

else:
    class PyTorchDeepUnfoldingNetwork:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is not installed in current environment. Use core.joint_pipeline.JointUnderwaterPipeline for full native execution.")
