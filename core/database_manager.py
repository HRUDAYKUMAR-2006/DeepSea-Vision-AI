r"""
Database Manager Module for Search Party Operations
===================================================
Maintains two separate, persistent SQLite databases:
1. storage/received_data.db: Storing incoming raw sensor feeds, GPS coordinates & mission metadata
2. storage/enhanced_results.db: Storing enhanced imagery, unrolled stages, detected targets & joint loss
"""

import os
import sqlite3
import json
import datetime
from typing import Dict, Any, List, Optional


class DatabaseManager:
    """
    Manages persistent storage for received underwater telemetry data
    and enhanced imagery records in two dedicated database files.
    """

    def __init__(self, base_storage_dir: str = "storage"):
        self.base_dir = base_storage_dir
        self.uploads_dir = os.path.join(self.base_dir, "uploads")
        self.enhanced_dir = os.path.join(self.base_dir, "enhanced")

        os.makedirs(self.uploads_dir, exist_ok=True)
        os.makedirs(self.enhanced_dir, exist_ok=True)

        self.db_received_path = os.path.join(self.base_dir, "received_data.db")
        self.db_enhanced_path = os.path.join(self.base_dir, "enhanced_results.db")

        self._init_databases()

    def _init_databases(self):
        """Initializes tables in both separate databases."""
        # 1. Received Data Database
        with sqlite3.connect(self.db_received_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS received_images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id TEXT UNIQUE NOT NULL,
                    filename TEXT,
                    timestamp TEXT NOT NULL,
                    depth_m REAL NOT NULL,
                    total_pressure_atm REAL NOT NULL,
                    pressure_kpa REAL NOT NULL,
                    safety_status TEXT NOT NULL,
                    light_zone TEXT NOT NULL,
                    ambient_lux REAL,
                    red_loss_pct REAL,
                    raw_uciqe REAL,
                    raw_uiqm REAL,
                    raw_image_path TEXT NOT NULL,
                    source_type TEXT DEFAULT 'uploaded',
                    mission_code TEXT DEFAULT 'SAR-OFFSHORE-01',
                    gps_coords TEXT DEFAULT '17.3850 N, 78.4867 E',
                    search_org TEXT DEFAULT 'Marine Search & Rescue Unit',
                    target_type TEXT DEFAULT 'all',
                    rov_unit_id TEXT DEFAULT 'ROV-EXPLORER-01'
                )
            """)
            # Migration check for existing DB files
            for col, col_type in [
                ("mission_code", "TEXT DEFAULT 'SAR-OFFSHORE-01'"),
                ("gps_coords", "TEXT DEFAULT '17.3850 N, 78.4867 E'"),
                ("search_org", "TEXT DEFAULT 'Marine Search & Rescue Unit'"),
                ("target_type", "TEXT DEFAULT 'all'"),
                ("rov_unit_id", "TEXT DEFAULT 'ROV-EXPLORER-01'")
            ]:
                try:
                    cursor.execute(f"ALTER TABLE received_images ADD COLUMN {col} {col_type}")
                except sqlite3.OperationalError:
                    pass
            conn.commit()

        # 2. Enhanced Results Database
        with sqlite3.connect(self.db_enhanced_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS enhanced_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    enhancement_id TEXT UNIQUE NOT NULL,
                    record_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    depth_m REAL NOT NULL,
                    enhanced_image_path TEXT NOT NULL,
                    stage_1_path TEXT,
                    stage_2_path TEXT,
                    stage_3_path TEXT,
                    transmission_map_path TEXT,
                    detection_overlay_path TEXT,
                    enhanced_uciqe REAL,
                    enhanced_uiqm REAL,
                    uciqe_gain REAL,
                    joint_loss_L REAL,
                    loss_rec REAL,
                    loss_perc REAL,
                    loss_det REAL,
                    loss_stage REAL,
                    detections_json TEXT,
                    num_detections INTEGER DEFAULT 0,
                    detected_classes TEXT,
                    mission_code TEXT DEFAULT 'SAR-OFFSHORE-01',
                    search_verdict TEXT DEFAULT 'ANALYZED'
                )
            """)
            for col, col_type in [
                ("mission_code", "TEXT DEFAULT 'SAR-OFFSHORE-01'"),
                ("search_verdict", "TEXT DEFAULT 'ANALYZED'")
            ]:
                try:
                    cursor.execute(f"ALTER TABLE enhanced_records ADD COLUMN {col} {col_type}")
                except sqlite3.OperationalError:
                    pass
            conn.commit()

    def save_received_record(
        self,
        record_id: str,
        filename: str,
        depth_m: float,
        telemetry: Dict[str, Any],
        raw_image_path: str,
        raw_uciqe: float = 0.0,
        raw_uiqm: float = 0.0,
        source_type: str = "uploaded",
        mission_code: str = "SAR-OFFSHORE-01",
        gps_coords: str = "17.3850 N, 78.4867 E",
        search_org: str = "Marine Search & Rescue Unit",
        target_type: str = "all",
        rov_unit_id: str = "ROV-EXPLORER-01"
    ) -> int:
        """Saves incoming image metadata into received_data.db."""
        now_iso = datetime.datetime.now().isoformat()
        pressure = telemetry.get("pressure", {})
        optics = telemetry.get("optics_light", {})

        with sqlite3.connect(self.db_received_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO received_images (
                    record_id, filename, timestamp, depth_m,
                    total_pressure_atm, pressure_kpa, safety_status,
                    light_zone, ambient_lux, red_loss_pct,
                    raw_uciqe, raw_uiqm, raw_image_path, source_type,
                    mission_code, gps_coords, search_org, target_type, rov_unit_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record_id,
                filename,
                now_iso,
                float(depth_m),
                float(pressure.get("total_pressure_atm", 1.0)),
                float(pressure.get("total_pressure_kpa", 101.3)),
                str(pressure.get("safety_status", "SAFE")),
                str(optics.get("zone", "Euphotic Zone")),
                float(optics.get("ambient_lux", 1000.0)),
                float(optics.get("red_loss_percent", 0.0)),
                float(raw_uciqe),
                float(raw_uiqm),
                raw_image_path,
                source_type,
                mission_code,
                gps_coords,
                search_org,
                target_type,
                rov_unit_id
            ))
            conn.commit()
            return cursor.lastrowid

    def save_enhanced_record(
        self,
        enhancement_id: str,
        record_id: str,
        depth_m: float,
        enhanced_image_path: str,
        stage_1_path: str,
        stage_2_path: str,
        stage_3_path: str,
        transmission_map_path: str,
        detection_overlay_path: str,
        metrics: Dict[str, Any],
        joint_loss: Optional[Dict[str, Any]],
        detections: List[Dict[str, Any]],
        mission_code: str = "SAR-OFFSHORE-01"
    ) -> int:
        """Saves enhancement and marine detection results into enhanced_results.db."""
        now_iso = datetime.datetime.now().isoformat()
        enh_uciqe = float(metrics.get("enhanced_uciqe", 0.0))
        deg_uciqe = float(metrics.get("degraded_uciqe", 0.0))
        uciqe_gain = enh_uciqe - deg_uciqe

        enh_uiqm = 0.0
        if isinstance(metrics.get("enhanced_uiqm"), dict):
            enh_uiqm = float(metrics["enhanced_uiqm"].get("UIQM", 0.0))
        elif isinstance(metrics.get("enhanced_uiqm"), (int, float)):
            enh_uiqm = float(metrics["enhanced_uiqm"])

        loss_L = joint_loss.get("total_loss_L", 0.0) if joint_loss else 0.0
        l_rec = joint_loss.get("reconstruction_loss_L_rec", 0.0) if joint_loss else 0.0
        l_perc = joint_loss.get("perceptual_loss_L_perc", 0.0) if joint_loss else 0.0
        l_det = joint_loss.get("detection_loss_L_det", 0.0) if joint_loss else 0.0
        l_stage = joint_loss.get("multi_stage_loss_L_stage", 0.0) if joint_loss else 0.0

        classes_list = [d.get("class_name", "") for d in detections]
        unique_classes = list(set([c for c in classes_list if c]))
        classes_str = ", ".join(unique_classes)
        verdict = f"TARGETS_CONFIRMED ({len(detections)})" if detections else "NO_TARGET_DETECTED"

        with sqlite3.connect(self.db_enhanced_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO enhanced_records (
                    enhancement_id, record_id, timestamp, depth_m,
                    enhanced_image_path, stage_1_path, stage_2_path, stage_3_path,
                    transmission_map_path, detection_overlay_path,
                    enhanced_uciqe, enhanced_uiqm, uciqe_gain,
                    joint_loss_L, loss_rec, loss_perc, loss_det, loss_stage,
                    detections_json, num_detections, detected_classes,
                    mission_code, search_verdict
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                enhancement_id,
                record_id,
                now_iso,
                float(depth_m),
                enhanced_image_path,
                stage_1_path,
                stage_2_path,
                stage_3_path,
                transmission_map_path,
                detection_overlay_path,
                enh_uciqe,
                enh_uiqm,
                uciqe_gain,
                loss_L,
                l_rec,
                l_perc,
                l_det,
                l_stage,
                json.dumps(detections),
                len(detections),
                classes_str,
                mission_code,
                verdict
            ))
            conn.commit()
            return cursor.lastrowid

    def get_all_records(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves combined history of processed records from both databases."""
        records = []
        with sqlite3.connect(self.db_received_path) as conn_r:
            conn_r.row_factory = sqlite3.Row
            cursor_r = conn_r.cursor()
            cursor_r.execute("SELECT * FROM received_images ORDER BY id DESC LIMIT ?", (limit,))
            received_rows = cursor_r.fetchall()

        enhanced_map = {}
        with sqlite3.connect(self.db_enhanced_path) as conn_e:
            conn_e.row_factory = sqlite3.Row
            cursor_e = conn_e.cursor()
            cursor_e.execute("SELECT * FROM enhanced_records ORDER BY id DESC")
            for row in cursor_e.fetchall():
                enhanced_map[row["record_id"]] = dict(row)

        for r_row in received_rows:
            r_dict = dict(r_row)
            rec_id = r_dict["record_id"]
            enh_dict = enhanced_map.get(rec_id, {})
            records.append({
                "record": r_dict,
                "enhancement": enh_dict
            })

        return records

    def get_record_by_id(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Fetches single combined record by ID."""
        with sqlite3.connect(self.db_received_path) as conn_r:
            conn_r.row_factory = sqlite3.Row
            cursor = conn_r.cursor()
            cursor.execute("SELECT * FROM received_images WHERE record_id = ?", (record_id,))
            r_row = cursor.fetchone()
            if not r_row:
                return None
            r_dict = dict(r_row)

        with sqlite3.connect(self.db_enhanced_path) as conn_e:
            conn_e.row_factory = sqlite3.Row
            cursor_e = conn_e.cursor()
            cursor_e.execute("SELECT * FROM enhanced_records WHERE record_id = ?", (record_id,))
            e_row = cursor_e.fetchone()
            e_dict = dict(e_row) if e_row else {}

        return {
            "record": r_dict,
            "enhancement": e_dict
        }
