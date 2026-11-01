r"""
Automated Test Suite for Detection-Aware Deep Unfolding Network (500M Edition)
==============================================================================
Validates:
1. Camera physics, light zones & hydrostatic pressure equations up to 500m:
      P(d) = P_0 + (\rho * g * d) / 101325
2. Optical degradation model & gradient calculation (Slide 13):
      I(x) = J(x) * t(x) + B * (1 - t(x))
      \nabla f(J) = - (I - (J*t + B(1-t))) * t
3. Unfolded K-stage progression (Slide 12: Stage 1, Stage 2, Stage 3)
4. Marine Object Detection & NMS
5. Joint end-to-end training loss (Slide 10):
      L = L_rec + \alpha * L_perc + \beta * L_det + \gamma * L_stage
6. Dual persistent database storage & AI Report generation
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np
import cv2

from core.camera_physics import UnderwaterCameraProfile
from core.image_formation import UnderwaterImageFormation, DegradationEstimator
from core.detector import MarineObjectDetector, DetectionBox
from core.deep_unfolding import DetectionAwareDeepUnfoldingNetwork
from core.metrics import compute_psnr, compute_ssim, compute_uciqe, compute_uiqm, evaluate_detections
from core.joint_pipeline import JointUnderwaterPipeline
from core.database_manager import DatabaseManager
from core.ai_report_generator import AIReportGenerator


def test_camera_pressure_equation_500m():
    cam = UnderwaterCameraProfile(
        min_depth_m=0.0,
        max_depth_m=500.0,
        surface_pressure_atm=1.0,
        water_density_kg_m3=1025.0,
        gravity_m_s2=9.80665,
        max_rated_pressure_atm=56.0
    )

    # 1. Surface check (d = 0m -> P = 1.0 atm)
    p0 = cam.compute_pressure(0.0)
    assert pytest.approx(p0["total_pressure_atm"], 0.01) == 1.0
    assert p0["safety_status"] == "SAFE"

    # 2. 200m depth (photic boundary): P = 1.0 + (1025 * 9.80665 * 200) / 101325 ≈ 20.84 atm
    p200 = cam.compute_pressure(200.0)
    assert pytest.approx(p200["total_pressure_atm"], 0.1) == 20.84
    assert p200["safety_status"] == "SAFE"

    # 3. 500m depth (deep-sea limit): P ≈ 50.60 atm -> approaches 56.0 atm (>90%) -> warning
    p500 = cam.compute_pressure(500.0)
    assert pytest.approx(p500["total_pressure_atm"], 0.1) == 50.60
    assert p500["safety_status"] == "WARNING_HIGH_PRESSURE"

    # 4. 560m depth (exceeds max 500m rating) -> out of range
    p560 = cam.compute_pressure(560.0)
    assert p560["safety_status"] == "OUT_OF_RANGE"
    assert not p560["is_valid_operating_zone"]


def test_optical_attenuation_and_mesopelagic_zone():
    cam = UnderwaterCameraProfile(max_depth_m=500.0)
    # At 350m: mesopelagic zone
    light_info = cam.compute_light_transmission(350.0)
    assert light_info["zone"] == "Mesopelagic (Twilight Zone)"
    assert light_info["red_loss_percent"] >= 99.0


def test_image_formation_and_gradient():
    formation = UnderwaterImageFormation(t_min=0.05, t_max=0.98)
    clean = np.ones((64, 64, 3), dtype=np.float32) * 0.8
    t = np.ones((64, 64, 3), dtype=np.float32) * 0.5
    B = np.array([0.1, 0.4, 0.6], dtype=np.float32)

    degraded = formation.forward_degradation(clean, t, B)
    expected = 0.4 + 0.5 * B.reshape(1, 1, 3)
    assert np.allclose(degraded, expected, atol=1e-5)

    grad_at_optimum = formation.compute_data_fidelity_gradient(clean, degraded, t, B)
    assert np.allclose(grad_at_optimum, 0.0, atol=1e-5)


def test_deep_unfolding_stages():
    detector = MarineObjectDetector()
    net = DetectionAwareDeepUnfoldingNetwork(num_stages=3)

    sample_input = np.ones((120, 120, 3), dtype=np.float32) * 0.4
    sample_input[:, :, 0] = 0.1
    sample_input[:, :, 1] = 0.45
    sample_input[:, :, 2] = 0.55

    res = net.forward(sample_input, detector)
    assert len(res["stage_outputs"]) == 3
    assert res["enhanced_image"].shape == (120, 120, 3)

    # Stage 1 red restoration
    s1 = res["stage_outputs"][0]
    assert s1[:, :, 0].mean() > sample_input[:, :, 0].mean()


def test_marine_detector():
    detector = MarineObjectDetector()
    img = np.zeros((200, 200, 3), dtype=np.float32)
    cv2.ellipse(img, (100, 100), (50, 20), 0, 0, 360, (0.85, 0.45, 0.1), -1)

    F_det = detector.extract_detector_guidance_features(img)
    assert F_det.shape == (200, 200, 32)

    detections = detector.detect(img)
    assert len(detections) >= 1
    assert detections[0].class_name in detector.classes


def test_joint_pipeline_and_database_storage():
    pipeline = JointUnderwaterPipeline()
    db = DatabaseManager(base_storage_dir="storage")

    img = np.ones((100, 100, 3), dtype=np.float32) * 0.5
    res = pipeline.process_frame(image_rgb=img, camera_depth_m=250.0)

    assert res["telemetry"]["pressure"]["total_pressure_atm"] > 20.0
    assert res["telemetry"]["optics_light"]["zone"] == "Mesopelagic (Twilight Zone)"

    # Test saving to dual databases
    rec_id = "TEST_REC_999"
    enh_id = "TEST_ENH_999"

    db.save_received_record(
        record_id=rec_id,
        filename="test.png",
        depth_m=250.0,
        telemetry=res["telemetry"],
        raw_image_path="test_raw.png",
        raw_uciqe=0.25
    )

    db.save_enhanced_record(
        enhancement_id=enh_id,
        record_id=rec_id,
        depth_m=250.0,
        enhanced_image_path="test_enh.png",
        stage_1_path="s1.png",
        stage_2_path="s2.png",
        stage_3_path="s3.png",
        transmission_map_path="t.png",
        detection_overlay_path="d.png",
        metrics=res["metrics"],
        joint_loss=res["joint_loss"],
        detections=res["enhanced_detections"]
    )

    fetched = db.get_record_by_id(rec_id)
    assert fetched is not None
    assert fetched["record"]["depth_m"] == 250.0
    assert fetched["enhancement"]["enhancement_id"] == enh_id


def test_ai_report_generation():
    dummy_rec = {
        "record_id": "REC_TEST_RPT",
        "timestamp": "2026-11-01T12:00:00",
        "depth_m": 450.0,
        "total_pressure_atm": 45.64,
        "pressure_kpa": 4624.5,
        "safety_status": "SAFE",
        "light_zone": "Mesopelagic (Twilight Zone)",
        "ambient_lux": 2.0,
        "red_loss_pct": 100.0,
        "raw_uciqe": 0.2450
    }
    dummy_enh = {
        "enhanced_uciqe": 0.3550,
        "joint_loss_L": 0.42,
        "loss_rec": 0.31,
        "loss_perc": 0.02,
        "loss_det": 0.001,
        "loss_stage": 0.12,
        "detections_json": '[{"class_name": "crustacean", "score": 0.96, "box": [100, 150, 250, 300]}]'
    }

    report = AIReportGenerator.generate_markdown_report(dummy_rec, dummy_enh)
    assert "MARINE SEARCH PARTY" in report
    assert "450.0 meters" in report
    assert "Crustacean" in report
    assert "0.3550" in report
