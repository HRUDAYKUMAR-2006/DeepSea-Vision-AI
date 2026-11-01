r"""
Main CLI Pipeline Runner
========================
Executes Detection-Aware Deep Unfolding and Marine Object Detection
on individual images or whole benchmark datasets.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import argparse
import json
import cv2
import numpy as np
from core.joint_pipeline import JointUnderwaterPipeline


def draw_bounding_boxes(image: np.ndarray, detections: list, color=(0, 255, 0)) -> np.ndarray:
    """Draws bounding boxes, labels, and confidence on an image."""
    vis = (image * 255).astype(np.uint8).copy()
    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        score = det["score"]
        cls_name = det["class_name"]

        # Box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        # Label banner
        label = f"{cls_name} {score:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, max(0, y1 - 20)), (x1 + w + 6, max(0, y1)), color, -1)
        cv2.putText(vis, label, (x1 + 3, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def main():
    parser = argparse.ArgumentParser(description="Detection-Aware Deep Unfolding Network for Joint Enhancement & Marine Detection")
    parser.add_argument("--image", type=str, default=None, help="Path to input degraded underwater image")
    parser.add_argument("--depth", type=float, default=18.0, help="Camera underwater depth in meters (0 - 50m)")
    parser.add_argument("--run-all-samples", action="store_true", help="Run benchmark evaluation across all sample scenarios")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save enhanced images and stage results")
    args = parser.parse_args()

    pipeline = JointUnderwaterPipeline()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.run_all_samples:
        manifest_path = "datasets/sample_images/dataset_manifest.json"
        if not os.path.exists(manifest_path):
            print("Dataset manifest not found. Generating sample dataset first...")
            from datasets.generate_dataset import generate_benchmark_dataset
            generate_benchmark_dataset()

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        print(f"\n{'='*75}")
        print(f"RUNNING BENCHMARK EVALUATION ON {len(manifest)} MARINE SCENARIOS")
        print(f"{'='*75}")

        results_summary = []

        for item in manifest:
            img_id = item["id"]
            depth = item["camera_depth_m"]
            degraded_path = item["degraded_image_path"]
            clean_path = item["clean_image_path"]
            gt_boxes = item["ground_truth_boxes"]

            degraded_bgr = cv2.imread(degraded_path)
            clean_bgr = cv2.imread(clean_path)
            degraded_rgb = cv2.cvtColor(degraded_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            result = pipeline.process_frame(
                image_rgb=degraded_rgb,
                camera_depth_m=depth,
                ground_truth_image=clean_rgb,
                ground_truth_boxes=gt_boxes
            )

            # Save stage progression
            scenario_out_dir = os.path.join(args.output_dir, img_id)
            os.makedirs(scenario_out_dir, exist_ok=True)

            cv2.imwrite(os.path.join(scenario_out_dir, "0_input_degraded.png"), degraded_bgr)
            cv2.imwrite(os.path.join(scenario_out_dir, "1_stage1_red_restored.png"), cv2.cvtColor((result["stage_1_color_fixed"] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(scenario_out_dir, "2_stage2_contrast_dehazed.png"), cv2.cvtColor((result["stage_2_contrast_haze"] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(scenario_out_dir, "3_stage3_edges_sharpened.png"), cv2.cvtColor((result["stage_3_edge_sharpened"] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

            # Detection visual overlay
            overlay_enhanced = draw_bounding_boxes(result["enhanced_image"], result["enhanced_detections"], color=(0, 255, 0))
            cv2.imwrite(os.path.join(scenario_out_dir, "4_final_detections.png"), cv2.cvtColor(overlay_enhanced, cv2.COLOR_RGB2BGR))

            m = result["metrics"]
            p_atm = result["telemetry"]["pressure"]["total_pressure_atm"]
            safety = result["telemetry"]["pressure"]["safety_status"]

            print(f"\nScenario: {img_id}")
            print(f" - Depth: {depth:.1f}m | Pressure: {p_atm:.2f} atm | Safety: {safety}")
            print(f" - PSNR: Raw = {m.get('psnr_degraded_to_gt', 0):.2f} dB -> Enhanced = {m.get('psnr_enhanced_to_gt', 0):.2f} dB (Gain: +{m.get('psnr_enhanced_to_gt', 0) - m.get('psnr_degraded_to_gt', 0):.2f} dB)")
            print(f" - SSIM: Raw = {m.get('ssim_degraded_to_gt', 0):.4f} -> Enhanced = {m.get('ssim_enhanced_to_gt', 0):.4f}")
            print(f" - UCIQE: Raw = {m['degraded_uciqe']:.4f} -> Enhanced = {m['enhanced_uciqe']:.4f}")
            print(f" - Detected Marine Objects: {len(result['enhanced_detections'])}")
            for d in result["enhanced_detections"]:
                print(f"     * {d['class_name'].capitalize()} (Confidence: {d['score']:.2f}) at Box: {d['box']}")
            if result["joint_loss"]:
                print(f" - Joint Loss L: {result['joint_loss']['total_loss_L']} (L_rec: {result['joint_loss']['reconstruction_loss_L_rec']}, L_det: {result['joint_loss']['detection_loss_L_det']}, L_stage: {result['joint_loss']['multi_stage_loss_L_stage']})")

            results_summary.append({
                "scenario": img_id,
                "depth_m": depth,
                "pressure_atm": p_atm,
                "safety_status": safety,
                "psnr_gain": m.get("psnr_enhanced_to_gt", 0) - m.get("psnr_degraded_to_gt", 0),
                "enhanced_ssim": m.get("ssim_enhanced_to_gt", 0),
                "detections_count": len(result["enhanced_detections"])
            })

        print(f"\n{'='*75}")
        print("ALL SCENARIOS PROCESSED SUCCESSFULLY!")
        print(f"Results saved to '{args.output_dir}/'")
        print(f"{'='*75}\n")

    else:
        # Single image inference
        img_path = args.image or "datasets/sample_images/sample_2_turtle_mid_degraded.png"
        if not os.path.exists(img_path):
            print(f"Image '{img_path}' not found. Generating sample dataset first...")
            from datasets.generate_dataset import generate_benchmark_dataset
            generate_benchmark_dataset()

        bgr = cv2.imread(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        print(f"Processing image: {img_path} at Depth {args.depth}m...")
        res = pipeline.process_frame(image_rgb=rgb, camera_depth_m=args.depth)

        p = res["telemetry"]["pressure"]
        print("\n--- CAMERA & ENVIRONMENTAL TELEMETRY ---")
        print(f"Atmospheric Pressure: {p['total_pressure_atm']:.2f} atm ({p['total_pressure_kpa']:.1f} kPa)")
        print(f"Operating Envelope Safety: {p['safety_status']}")
        if p["warning"]:
            print(f"WARNING: {p['warning']}")

        print("\n--- DEEP UNFOLDING STAGE RESULTS ---")
        print(f"Stage 1 (Color cast fixed / Red restored) Mean: {res['stage_1_color_fixed'].mean():.4f}")
        print(f"Stage 2 (Haze removed / Contrast raised) Mean: {res['stage_2_contrast_haze'].mean():.4f}")
        print(f"Stage 3 (Edges sharpened / Detector guidance) Mean: {res['stage_3_edge_sharpened'].mean():.4f}")

        print("\n--- MARINE OBJECT DETECTIONS ---")
        print(f"Total Objects Detected: {len(res['enhanced_detections'])}")
        for d in res['enhanced_detections']:
            print(f" - {d['class_name'].upper()}: Confidence = {d['score']:.2f}, Box = {d['box']}")

        out_file = os.path.join(args.output_dir, "enhanced_detection_result.png")
        overlay = draw_bounding_boxes(res["enhanced_image"], res["enhanced_detections"])
        cv2.imwrite(out_file, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"\nEnhanced output image saved to '{out_file}'")


if __name__ == "__main__":
    main()

