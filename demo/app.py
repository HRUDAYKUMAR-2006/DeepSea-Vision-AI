r"""
Web Application & Interactive Dashboard (Search Party & Standalone App Edition)
==============================================================================
Flask-powered interactive dashboard & standalone application engine:
- Camera Operational Envelope adapted to 500m depth & 55+ atm hydrostatic pressure
- Dual persistent database storage (received_data.db & enhanced_results.db)
- Progressive Web App (PWA) manifest & service worker for 1-click desktop app install
- Direct zip bundle download for search party deployment
- Search & Rescue Mission Mode with GPS coordinates, mission tags, and CSV export
- Instant AI Inspection Report generation
- Light, clean, bright modern UI theme
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import io
import csv
import time
import base64
import json
import uuid
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory, Response

from core.joint_pipeline import JointUnderwaterPipeline
from core.database_manager import DatabaseManager
from core.ai_report_generator import AIReportGenerator


app = Flask(__name__, template_folder="templates", static_folder="static")
pipeline = JointUnderwaterPipeline()
db_manager = DatabaseManager(base_storage_dir="storage")

DIST_ZIP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dist", "DeepSea_Vision_AI_SearchParty_Edition.zip"))


def ndarray_to_base64_png(img_rgb: np.ndarray) -> str:
    """Converts a float32 [0, 1] RGB ndarray to base64 PNG data URL."""
    img_uint8 = (np.clip(img_rgb, 0.0, 1.0) * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', img_bgr)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{b64_str}"


def draw_detection_overlay(img_rgb: np.ndarray, detections: list) -> np.ndarray:
    """Draws boxes, labels and confidence on top of RGB image."""
    vis = (np.clip(img_rgb, 0.0, 1.0) * 255).astype(np.uint8).copy()
    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

    color_palette = {
        "fish": (16, 185, 129),        # Emerald green
        "turtle": (245, 158, 11),      # Amber
        "echinoderm": (168, 85, 247),  # Purple
        "crustacean": (239, 68, 68)    # Coral red
    }

    for det in detections:
        box = det["box"]
        x1, y1, x2, y2 = map(int, box)
        cls_name = det["class_name"]
        score = det["score"]
        col = color_palette.get(cls_name, (59, 130, 246))

        # Bounding Box
        cv2.rectangle(vis_bgr, (x1, y1), (x2, y2), col, 2)

        # Label tag
        label = f"{cls_name.capitalize()} {score:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis_bgr, (x1, max(0, y1 - 22)), (x1 + w + 10, max(0, y1)), col, -1)
        cv2.putText(vis_bgr, label, (x1 + 5, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


@app.route("/")
def index():
    manifest_path = "datasets/sample_images/dataset_manifest.json"
    scenarios = []
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            scenarios = json.load(f)
    return render_template("index.html", scenarios=scenarios)


@app.route("/sw.js")
def service_worker():
    """Serves PWA service worker from root scope."""
    return send_from_directory("static", "sw.js", mimetype="application/javascript")


@app.route("/manifest.json")
def pwa_manifest():
    """Serves web app manifest."""
    return send_from_directory("static", "manifest.json", mimetype="application/manifest+json")


@app.route("/api/download/app-bundle")
def download_app_bundle():
    """Provides direct download of standalone search party application zip."""
    if not os.path.exists(DIST_ZIP_PATH):
        # Repackage if missing
        import scripts.package_app as pkg
        pkg.package()
    return send_file(
        DIST_ZIP_PATH,
        as_attachment=True,
        download_name="DeepSea_Vision_AI_SearchParty_Edition.zip",
        mimetype="application/zip"
    )


@app.route("/api/process", methods=["POST"])
def process_api():
    try:
        data = request.json or {}
        scenario_id = data.get("scenario_id")
        depth_m = float(data.get("depth_m", 150.0))
        custom_image_b64 = data.get("custom_image_b64")
        filename = data.get("filename", "sensor_capture.png")

        # Search party metadata
        mission_code = data.get("mission_code", "SAR-OFFSHORE-01")
        gps_coords = data.get("gps_coords", "17.3850 N, 78.4867 E")
        search_org = data.get("search_org", "Marine Search & Rescue Unit")
        target_type = data.get("target_type", "all")
        rov_unit_id = data.get("rov_unit_id", "ROV-EXPLORER-01")

        img_rgb = None
        gt_rgb = None
        gt_boxes = None
        source_type = "scenario"

        if custom_image_b64:
            source_type = "uploaded"
            header, encoded = custom_image_b64.split(",", 1) if "," in custom_image_b64 else ("", custom_image_b64)
            img_bytes = base64.b64decode(encoded)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                return jsonify({"error": "Failed to decode uploaded image"}), 400
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        elif scenario_id:
            manifest_path = "datasets/sample_images/dataset_manifest.json"
            if os.path.exists(manifest_path):
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                match = next((item for item in manifest if item["id"] == scenario_id), None)
                if match:
                    filename = f"{scenario_id}.png"
                    deg_bgr = cv2.imread(match["degraded_image_path"])
                    clean_bgr = cv2.imread(match["clean_image_path"])
                    if deg_bgr is not None:
                        img_rgb = cv2.cvtColor(deg_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    if clean_bgr is not None:
                        gt_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    gt_boxes = match.get("ground_truth_boxes")

        if img_rgb is None:
            fallback_path = "datasets/sample_images/sample_1_shallow_reef_degraded.png"
            if os.path.exists(fallback_path):
                bgr = cv2.imread(fallback_path)
                img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            else:
                return jsonify({"error": "No image found"}), 404

        # Run pipeline
        res = pipeline.process_frame(
            image_rgb=img_rgb,
            camera_depth_m=depth_m,
            ground_truth_image=gt_rgb,
            ground_truth_boxes=gt_boxes
        )

        # Unique Record & Enhancement IDs
        ts_slug = time.strftime("%Y%m%d_%H%M%S")
        rand_slug = uuid.uuid4().hex[:6]
        record_id = f"REC_{ts_slug}_{rand_slug}"
        enhancement_id = f"ENH_{ts_slug}_{rand_slug}"

        # Save files to disk for persistent storage
        raw_save_path = os.path.join(db_manager.uploads_dir, f"{record_id}_raw.png")
        enh_save_path = os.path.join(db_manager.enhanced_dir, f"{enhancement_id}_enhanced.png")
        s1_save_path = os.path.join(db_manager.enhanced_dir, f"{enhancement_id}_stage1.png")
        s2_save_path = os.path.join(db_manager.enhanced_dir, f"{enhancement_id}_stage2.png")
        s3_save_path = os.path.join(db_manager.enhanced_dir, f"{enhancement_id}_stage3.png")
        trans_save_path = os.path.join(db_manager.enhanced_dir, f"{enhancement_id}_trans.png")
        det_save_path = os.path.join(db_manager.enhanced_dir, f"{enhancement_id}_detections.png")

        overlay = draw_detection_overlay(res["enhanced_image"], res["enhanced_detections"])
        raw_overlay = draw_detection_overlay(res["raw_image"], res["raw_detections"])

        cv2.imwrite(raw_save_path, cv2.cvtColor((res["raw_image"] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(enh_save_path, cv2.cvtColor((res["enhanced_image"] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(s1_save_path, cv2.cvtColor((res["stage_1_color_fixed"] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(s2_save_path, cv2.cvtColor((res["stage_2_contrast_haze"] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(s3_save_path, cv2.cvtColor((res["stage_3_edge_sharpened"] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(trans_save_path, cv2.cvtColor((res["transmission_map"] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(det_save_path, cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

        # 1. Save into Received Database (storage/received_data.db)
        db_manager.save_received_record(
            record_id=record_id,
            filename=filename,
            depth_m=depth_m,
            telemetry=res["telemetry"],
            raw_image_path=raw_save_path,
            raw_uciqe=res["metrics"].get("degraded_uciqe", 0.0),
            raw_uiqm=res["metrics"].get("degraded_uiqm", {}).get("UIQM", 0.0) if isinstance(res["metrics"].get("degraded_uiqm"), dict) else 0.0,
            source_type=source_type,
            mission_code=mission_code,
            gps_coords=gps_coords,
            search_org=search_org,
            target_type=target_type,
            rov_unit_id=rov_unit_id
        )

        # 2. Save into Enhanced Database (storage/enhanced_results.db)
        db_manager.save_enhanced_record(
            enhancement_id=enhancement_id,
            record_id=record_id,
            depth_m=depth_m,
            enhanced_image_path=enh_save_path,
            stage_1_path=s1_save_path,
            stage_2_path=s2_save_path,
            stage_3_path=s3_save_path,
            transmission_map_path=trans_save_path,
            detection_overlay_path=det_save_path,
            metrics=res["metrics"],
            joint_loss=res["joint_loss"],
            detections=res["enhanced_detections"],
            mission_code=mission_code
        )

        response_payload = {
            "record_id": record_id,
            "enhancement_id": enhancement_id,
            "mission_code": mission_code,
            "gps_coords": gps_coords,
            "telemetry": res["telemetry"],
            "metrics": res["metrics"],
            "joint_loss": res["joint_loss"],
            "detections": res["enhanced_detections"],
            "raw_detections": res["raw_detections"],
            "images": {
                "input_degraded": ndarray_to_base64_png(res["raw_image"]),
                "transmission_map": ndarray_to_base64_png(res["transmission_map"]),
                "stage_1_color_fixed": ndarray_to_base64_png(res["stage_1_color_fixed"]),
                "stage_2_contrast_haze": ndarray_to_base64_png(res["stage_2_contrast_haze"]),
                "stage_3_edge_sharpened": ndarray_to_base64_png(res["stage_3_edge_sharpened"]),
                "enhanced_image": ndarray_to_base64_png(res["enhanced_image"]),
                "detection_overlay": ndarray_to_base64_png(overlay),
                "raw_detection_overlay": ndarray_to_base64_png(raw_overlay)
            }
        }
        return jsonify(response_payload)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/report", methods=["POST"])
def generate_report_api():
    """Generates comprehensive AI report for chosen record ID."""
    try:
        data = request.json or {}
        record_id = data.get("record_id")
        if not record_id:
            return jsonify({"error": "Missing record_id"}), 400

        full_record = db_manager.get_record_by_id(record_id)
        if not full_record or not full_record.get("record"):
            return jsonify({"error": f"Record '{record_id}' not found in database"}), 404

        rec = full_record["record"]
        enh = full_record.get("enhancement", {})

        report_markdown = AIReportGenerator.generate_markdown_report(rec, enh)

        return jsonify({
            "record_id": record_id,
            "report_markdown": report_markdown,
            "meta": {
                "depth_m": rec["depth_m"],
                "pressure_atm": rec["total_pressure_atm"],
                "safety_status": rec["safety_status"],
                "mission_code": rec.get("mission_code", "SAR-OFFSHORE-01"),
                "gps_coords": rec.get("gps_coords", ""),
                "num_detections": enh.get("num_detections", 0)
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/history", methods=["GET"])
def history_api():
    """Returns stored database entries from both tables."""
    try:
        records = db_manager.get_all_records(limit=50)
        return jsonify({"records": records})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/export-csv", methods=["GET"])
def export_csv_api():
    """Exports all database records as CSV for search party mission logging."""
    try:
        records = db_manager.get_all_records(limit=100)
        si = io.StringIO()
        cw = csv.writer(si)
        cw.writerow([
            "Record ID", "Mission Code", "Search Org", "GPS Coordinates", "Timestamp",
            "Depth (m)", "Total Pressure (atm)", "Pressure (kPa)", "Safety Status",
            "Light Zone", "Raw UCIQE", "Enhanced UCIQE", "UCIQE Gain",
            "Targets Detected", "Detected Classes", "Joint Loss L"
        ])
        for item in records:
            r = item["record"]
            e = item.get("enhancement", {})
            cw.writerow([
                r.get("record_id", ""),
                r.get("mission_code", ""),
                r.get("search_org", ""),
                r.get("gps_coords", ""),
                r.get("timestamp", ""),
                r.get("depth_m", ""),
                r.get("total_pressure_atm", ""),
                r.get("pressure_kpa", ""),
                r.get("safety_status", ""),
                r.get("light_zone", ""),
                r.get("raw_uciqe", ""),
                e.get("enhanced_uciqe", ""),
                e.get("uciqe_gain", ""),
                e.get("num_detections", 0),
                e.get("detected_classes", ""),
                e.get("joint_loss_L", "")
            ])
        output = si.getvalue()
        return Response(
            output,
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename=DeepSea_Search_Party_Log.csv"}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting DeepSea Vision AI Platform & Search Party Server...")
    print("Open http://127.0.0.1:5000 in your browser.")
    app.run(host="127.0.0.1", port=5000, debug=False)
