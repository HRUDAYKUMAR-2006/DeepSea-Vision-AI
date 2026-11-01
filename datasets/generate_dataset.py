r"""
Underwater Dataset & Marine Test Sample Generator (0m - 500m Operational Envelope)
==================================================================================
Generates photic to mesopelagic marine test scenes adhering strictly to:
1. Physical Camera & Hydrostatic Pressure Envelope:
      P(d) = P_0 + (\rho * g * d) / 101325
2. Optical Degradation Model (Slide 13):
      I(x) = J(x) * t(x) + B * (1 - t(x))
3. Operational depth coverage up to 500 meters
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import cv2
import json
from typing import List, Dict, Any, Tuple
from core.camera_physics import UnderwaterCameraProfile
from core.image_formation import UnderwaterImageFormation


def create_clean_marine_scene(
    scene_type: str = "coral_reef",
    width: int = 640,
    height: int = 480
) -> Tuple[np.ndarray, List[Dict[str, Any]], np.ndarray]:
    """Creates a clean marine ground truth scene with marine organisms."""
    img = np.zeros((height, width, 3), dtype=np.float32)

    for y in range(height):
        factor = y / float(height)
        top_color = np.array([0.15, 0.65, 0.85], dtype=np.float32)
        bottom_color = np.array([0.05, 0.35, 0.65], dtype=np.float32)
        img[y, :] = (1.0 - factor) * top_color + factor * bottom_color

    seabed_top = int(height * 0.72)
    for y in range(seabed_top, height):
        sand_ratio = (y - seabed_top) / (height - seabed_top)
        sand_color = np.array([0.76, 0.68, 0.50], dtype=np.float32)
        img[y, :] = img[y, :] * (1.0 - sand_ratio * 0.7) + sand_color * (sand_ratio * 0.7)

    cv2.circle(img, (int(width * 0.15), height - 30), 65, (0.85, 0.25, 0.20), -1)
    cv2.circle(img, (int(width * 0.22), height - 25), 45, (0.90, 0.45, 0.15), -1)
    cv2.circle(img, (int(width * 0.82), height - 35), 75, (0.80, 0.30, 0.35), -1)
    cv2.circle(img, (int(width * 0.88), height - 20), 55, (0.92, 0.60, 0.10), -1)

    depth_map = np.ones((height, width), dtype=np.float32) * 22.0
    for y in range(height):
        if y >= seabed_top:
            depth_map[y, :] = 4.0 + (1.0 - (y - seabed_top) / (height - seabed_top)) * 6.0
        else:
            depth_map[y, :] = 12.0 + (1.0 - y / float(height)) * 12.0

    boxes = []

    if scene_type in ["coral_reef", "fish_shallow"]:
        # Orange clownfish
        fx, fy, fw, fh = int(width * 0.42), int(height * 0.42), 120, 60
        cv2.ellipse(img, (fx, fy), (fw // 2, fh // 2), 0, 0, 360, (0.95, 0.42, 0.08), -1)
        cv2.line(img, (fx - 15, fy - fh // 2 + 5), (fx - 15, fy + fh // 2 - 5), (0.98, 0.98, 0.98), 8)
        cv2.line(img, (fx + 15, fy - fh // 2 + 10), (fx + 15, fy + fh // 2 - 10), (0.98, 0.98, 0.98), 8)
        pts_tail = np.array([[fx - fw // 2, fy], [fx - fw // 2 - 30, fy - 25], [fx - fw // 2 - 30, fy + 25]], np.int32)
        cv2.fillPoly(img, [pts_tail], (0.95, 0.35, 0.05))
        cv2.circle(img, (fx + fw // 2 - 20, fy - 8), 5, (0.1, 0.1, 0.1), -1)
        cv2.circle(img, (fx + fw // 2 - 19, fy - 9), 2, (1.0, 1.0, 1.0), -1)

        x1 = fx - fw // 2 - 32
        y1 = fy - fh // 2 - 10
        x2 = fx + fw // 2 + 5
        y2 = fy + fh // 2 + 10
        boxes.append({"box": [x1, y1, x2, y2], "class_name": "fish"})
        depth_map[y1:y2, x1:x2] = 5.5

        # Yellow tang fish
        fx2, fy2, fw2, fh2 = int(width * 0.70), int(height * 0.28), 75, 45
        cv2.ellipse(img, (fx2, fy2), (fw2 // 2, fh2 // 2), -10, 0, 360, (0.95, 0.85, 0.10), -1)
        pts_tail2 = np.array([[fx2 - fw2 // 2, fy2], [fx2 - fw2 // 2 - 18, fy2 - 15], [fx2 - fw2 // 2 - 18, fy2 + 15]], np.int32)
        cv2.fillPoly(img, [pts_tail2], (0.90, 0.75, 0.08))
        cv2.circle(img, (fx2 + fw2 // 2 - 12, fy2 - 5), 4, (0.1, 0.1, 0.1), -1)

        bx1, by1 = fx2 - fw2 // 2 - 20, fy2 - fh2 // 2 - 5
        bx2, by2 = fx2 + fw2 // 2 + 5, fy2 + fh2 // 2 + 5
        boxes.append({"box": [bx1, by1, bx2, by2], "class_name": "fish"})
        depth_map[by1:by2, bx1:bx2] = 11.0

    elif scene_type in ["turtle_medium", "mesopelagic_turtle"]:
        # Sea Turtle
        tx, ty, tw, th = int(width * 0.48), int(height * 0.45), 160, 110
        cv2.ellipse(img, (tx, ty), (tw // 2, th // 2), 15, 0, 360, (0.55, 0.42, 0.22), -1)
        cv2.ellipse(img, (tx, ty), (tw // 2 - 10, th // 2 - 10), 15, 0, 360, (0.65, 0.50, 0.28), 3)
        cv2.ellipse(img, (tx + tw // 2 + 20, ty - 15), (22, 16), 25, 0, 360, (0.45, 0.55, 0.32), -1)
        cv2.circle(img, (tx + tw // 2 + 28, ty - 18), 3, (0.1, 0.1, 0.1), -1)
        cv2.ellipse(img, (tx + 15, ty - th // 2 - 20), (45, 18), 45, 0, 360, (0.45, 0.55, 0.32), -1)
        cv2.ellipse(img, (tx - 25, ty + th // 2 + 15), (35, 14), -35, 0, 360, (0.45, 0.55, 0.32), -1)

        x1 = tx - tw // 2 - 15
        y1 = ty - th // 2 - 40
        x2 = tx + tw // 2 + 45
        y2 = ty + th // 2 + 35
        boxes.append({"box": [x1, y1, x2, y2], "class_name": "turtle"})
        depth_map[y1:y2, x1:x2] = 7.0

        # Echinoderm
        ex, ey, er = int(width * 0.25), int(height * 0.88), 28
        cv2.circle(img, (ex, ey), er, (0.15, 0.08, 0.18), -1)
        for angle in range(0, 360, 20):
            rad = np.radians(angle)
            sx = int(ex + (er + 15) * np.cos(rad))
            sy = int(ey + (er + 15) * np.sin(rad))
            cv2.line(img, (ex, ey), (sx, sy), (0.1, 0.05, 0.12), 2)
        boxes.append({"box": [ex - er - 16, ey - er - 16, ex + er + 16, ey + er + 16], "class_name": "echinoderm"})
        depth_map[ey - er - 16:ey + er + 16, ex - er - 16:ex + er + 16] = 3.5

    elif scene_type in ["deep_photic_crustacean", "deep_500m_crab"]:
        # Deep Crab / Crustacean
        cx, cy, cw, ch = int(width * 0.52), int(height * 0.82), 100, 65
        cv2.ellipse(img, (cx, cy), (cw // 2, ch // 2), 0, 0, 360, (0.85, 0.30, 0.15), -1)
        cv2.circle(img, (cx - cw // 2 - 15, cy - 15), 18, (0.88, 0.32, 0.16), -1)
        cv2.circle(img, (cx + cw // 2 + 15, cy - 15), 18, (0.88, 0.32, 0.16), -1)
        for offset in [-25, -5, 15]:
            cv2.line(img, (cx - cw // 2 + 5, cy + offset), (cx - cw // 2 - 25, cy + offset + 15), (0.75, 0.25, 0.12), 3)
            cv2.line(img, (cx + cw // 2 - 5, cy + offset), (cx + cw // 2 + 25, cy + offset + 15), (0.75, 0.25, 0.12), 3)
        boxes.append({"box": [cx - cw // 2 - 35, cy - ch // 2 - 10, cx + cw // 2 + 35, cy + ch // 2 + 25], "class_name": "crustacean"})
        depth_map[cy - ch // 2 - 10:cy + ch // 2 + 25, cx - cw // 2 - 35:cx + cw // 2 + 35] = 4.0

        # Deep silver fish
        fx, fy, fw, fh = int(width * 0.35), int(height * 0.35), 130, 50
        cv2.ellipse(img, (fx, fy), (fw // 2, fh // 2), -5, 0, 360, (0.75, 0.82, 0.90), -1)
        pts_tail = np.array([[fx - fw // 2, fy], [fx - fw // 2 - 25, fy - 20], [fx - fw // 2 - 25, fy + 20]], np.int32)
        cv2.fillPoly(img, [pts_tail], (0.65, 0.75, 0.85))
        cv2.circle(img, (fx + fw // 2 - 20, fy - 5), 4, (0.1, 0.1, 0.1), -1)
        boxes.append({"box": [fx - fw // 2 - 28, fy - fh // 2 - 5, fx + fw // 2 + 5, fy + fh // 2 + 5], "class_name": "fish"})
        depth_map[fy - fh // 2 - 5:fy + fh // 2 + 5, fx - fw // 2 - 28:fx + fw // 2 + 5] = 14.0

    return img, boxes, depth_map


def generate_benchmark_dataset(output_dir: str = "datasets/sample_images"):
    """Synthesizes degraded underwater samples up to 500m depth."""
    os.makedirs(output_dir, exist_ok=True)
    cam = UnderwaterCameraProfile(max_depth_m=500.0, max_rated_pressure_atm=56.0)
    formation = UnderwaterImageFormation()

    test_scenarios = [
        {
            "id": "sample_1_shallow_reef",
            "scene": "coral_reef",
            "camera_depth_m": 25.0,
            "description": "Shallow photic reef (25m depth, 3.48 atm). Moderate red attenuation."
        },
        {
            "id": "sample_2_turtle_pelagic",
            "scene": "turtle_medium",
            "camera_depth_m": 120.0,
            "description": "Mid-photic pelagic zone (120m depth, 12.90 atm). Pronounced blue-green cast."
        },
        {
            "id": "sample_3_twilight_crustacean",
            "scene": "deep_photic_crustacean",
            "camera_depth_m": 280.0,
            "description": "Mesopelagic twilight zone (280m depth, 28.77 atm). Low ambient light & turbidity."
        },
        {
            "id": "sample_4_deep_500m_survey",
            "scene": "deep_500m_crab",
            "camera_depth_m": 500.0,
            "description": "Rated 500m deep-sea limit (500m depth, 50.60 atm). Approaching maximum housing pressure."
        },
        {
            "id": "sample_5_overpressure_560m",
            "scene": "turtle_medium",
            "camera_depth_m": 560.0,
            "description": "Overpressure threshold test (560m depth, 56.55 atm). Exceeds 500m rating and triggers safety alert."
        }
    ]

    manifest = []

    for sc in test_scenarios:
        sc_id = sc["id"]
        depth = sc["camera_depth_m"]
        telemetry = cam.get_telemetry_report(depth)

        clean_img, boxes, depth_map = create_clean_marine_scene(sc["scene"])

        beta_r, beta_g, beta_b = cam.attenuation_rgb
        depth_scale = 1.0 + min(depth / 50.0, 4.0)
        t_r = np.exp(-beta_r * (depth_map / 3.0) * depth_scale)
        t_g = np.exp(-beta_g * (depth_map / 3.0) * depth_scale)
        t_b = np.exp(-beta_b * (depth_map / 3.0) * depth_scale)
        transmission = np.stack([t_r, t_g, t_b], axis=2)
        transmission = np.clip(transmission, 0.05, 0.95)

        B = np.array([
            0.08 + 0.04 * min(depth / 500.0, 1.0),
            0.35 + 0.15 * min(depth / 500.0, 1.0),
            0.55 + 0.18 * min(depth / 500.0, 1.0)
        ], dtype=np.float32)
        B = np.clip(B, 0.05, 0.95)

        degraded_img = formation.forward_degradation(clean_img, transmission, B)
        noise = np.random.normal(0, 0.012, degraded_img.shape).astype(np.float32)
        degraded_img = np.clip(degraded_img + noise, 0.0, 1.0)

        clean_path = os.path.join(output_dir, f"{sc_id}_clean.png")
        degraded_path = os.path.join(output_dir, f"{sc_id}_degraded.png")

        cv2.imwrite(clean_path, cv2.cvtColor((clean_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(degraded_path, cv2.cvtColor((degraded_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

        entry = {
            "id": sc_id,
            "description": sc["description"],
            "camera_depth_m": depth,
            "clean_image_path": clean_path,
            "degraded_image_path": degraded_path,
            "ground_truth_boxes": boxes,
            "telemetry": telemetry,
            "background_light_gt": B.tolist()
        }
        manifest.append(entry)

    manifest_path = os.path.join(output_dir, "dataset_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated {len(manifest)} 500m marine scenarios in '{output_dir}'.")
    return manifest


if __name__ == "__main__":
    generate_benchmark_dataset()
