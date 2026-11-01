r"""
Camera Physics & Underwater Environmental Constraints Module
===========================================================
Defines the operational envelope for deep-sea underwater imaging sensors:
- Operational depth: Sea level surface (0m) to mesopelagic rating (500m)
- Hydrostatic & Underwater Atmospheric Pressure:
    P(d) = P_0 + (rho * g * d) / 101325   [atm]
- Light Attenuation & Zone Transitions:
    0m - 200m: Euphotic (Sunlight Zone)
    200m - 500m: Mesopelagic (Twilight Zone)
    > 500m: Bathypelagic / Aphotic (Midnight Zone)
"""

import math
from typing import Dict, Any, Tuple


class UnderwaterCameraProfile:
    """
    Simulates and validates physical camera operational parameters under
    extended marine hydrostatic pressure and light regions up to 500m depth.
    """
    def __init__(
        self,
        camera_name: str = "DeepExplorer-Pro-500M",
        min_depth_m: float = 0.0,
        max_depth_m: float = 500.0,
        surface_pressure_atm: float = 1.0,
        water_density_kg_m3: float = 1025.0,  # Seawater standard
        gravity_m_s2: float = 9.80665,
        max_rated_pressure_atm: float = 56.0,  # Rated housing limit (~56 atm / ~5674 kPa)
        attenuation_coefficients: Tuple[float, float, float] = (0.350, 0.080, 0.050) # R, G, B in m^-1
    ):
        self.camera_name = camera_name
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m
        self.surface_pressure_atm = surface_pressure_atm
        self.water_density = water_density_kg_m3
        self.gravity = gravity_m_s2
        self.max_rated_pressure_atm = max_rated_pressure_atm
        self.attenuation_rgb = attenuation_coefficients  # beta_R, beta_G, beta_B

    def compute_pressure(self, depth_m: float) -> Dict[str, Any]:
        """
        Calculates underwater hydrostatic pressure and total atmospheric/hydrostatic pressure.
        Formula:
            P_hydro(d) = rho * g * d  (Pascals)
            P_total(d) = P_surface + P_hydro(d) / 101325  (atm)
        At 500m: P_total = 1.0 + (1025 * 9.80665 * 500) / 101325 ≈ 50.61 atm (5128.5 kPa)
        """
        d = max(0.0, float(depth_m))
        hydrostatic_pa = self.water_density * self.gravity * d
        hydrostatic_atm = hydrostatic_pa / 101325.0
        total_pressure_atm = self.surface_pressure_atm + hydrostatic_atm
        total_pressure_kpa = total_pressure_atm * 101.325

        # Safety & rating envelope validation
        in_depth_range = (self.min_depth_m <= d <= self.max_depth_m)
        pressure_safe = (total_pressure_atm <= self.max_rated_pressure_atm)
        safety_status = "SAFE"
        warning_msg = None

        if not in_depth_range:
            safety_status = "OUT_OF_RANGE"
            warning_msg = (
                f"Depth {d:.1f}m exceeds the camera's maximum depth rating "
                f"[{self.min_depth_m}m - {self.max_depth_m}m]."
            )
        elif not pressure_safe:
            safety_status = "PRESSURE_LIMIT_EXCEEDED"
            warning_msg = (
                f"Pressure {total_pressure_atm:.2f} atm exceeds maximum housing "
                f"rating ({self.max_rated_pressure_atm:.2f} atm)!"
            )
        elif total_pressure_atm >= 0.90 * self.max_rated_pressure_atm:
            safety_status = "WARNING_HIGH_PRESSURE"
            warning_msg = f"Pressure approaching maximum rated limit: {total_pressure_atm:.2f} atm (>90% rated)."

        return {
            "depth_m": d,
            "hydrostatic_pressure_atm": hydrostatic_atm,
            "total_pressure_atm": total_pressure_atm,
            "total_pressure_kpa": total_pressure_kpa,
            "safety_status": safety_status,
            "is_valid_operating_zone": in_depth_range and pressure_safe,
            "warning": warning_msg
        }

    def compute_light_transmission(self, depth_m: float) -> Dict[str, Any]:
        """
        Calculates downwelling spectral transmission and light zone classification up to 500m.
        """
        d = max(0.0, float(depth_m))
        beta_r, beta_g, beta_b = self.attenuation_rgb
        trans_r = math.exp(-beta_r * min(d, 50.0))
        trans_g = math.exp(-beta_g * min(d, 200.0))
        trans_b = math.exp(-beta_b * min(d, 350.0))

        # Photic Zone Classification up to 500m
        if d <= 200.0:
            zone = "Euphotic (Sunlight Zone)"
            lighting_type = "Natural Solar Downwelling"
        elif d <= 500.0:
            zone = "Mesopelagic (Twilight Zone)"
            lighting_type = "Low Ambient + ROV Floodlights"
        else:
            zone = "Bathypelagic / Aphotic (Midnight Zone)"
            lighting_type = "Complete Darkness (Artificial Illumination Required)"

        surface_lux = 100000.0
        ambient_lux = max(surface_lux * math.exp(-0.045 * d), 0.05)
        red_loss = min(100.0, (1.0 - math.exp(-beta_r * d)) * 100.0)

        return {
            "depth_m": d,
            "zone": zone,
            "lighting_type": lighting_type,
            "ambient_lux": ambient_lux,
            "transmission_rgb": {
                "red": trans_r,
                "green": trans_g,
                "blue": trans_b
            },
            "red_loss_percent": red_loss,
            "color_cast_tendency": "Deep Oceanic Blue / Backscatter" if d > 150 else ("Blue-Green" if d > 10 else "Natural")
        }

    def get_telemetry_report(self, depth_m: float) -> Dict[str, Any]:
        """Returns consolidated camera operating telemetry."""
        p_info = self.compute_pressure(depth_m)
        l_info = self.compute_light_transmission(depth_m)
        return {
            "camera_model": self.camera_name,
            "depth_meters": float(depth_m),
            "pressure": p_info,
            "optics_light": l_info,
            "operational_valid": p_info["is_valid_operating_zone"]
        }


if __name__ == "__main__":
    cam = UnderwaterCameraProfile()
    for test_d in [0.0, 50.0, 200.0, 350.0, 500.0, 560.0]:
        report = cam.get_telemetry_report(test_d)
        p = report["pressure"]
        l = report["optics_light"]
        print(f"\n--- Depth {test_d}m ---")
        print(f"Total Pressure: {p['total_pressure_atm']:.2f} atm ({p['total_pressure_kpa']:.1f} kPa)")
        print(f"Safety: {p['safety_status']} | Zone: {l['zone']}")
