"""Parametric component catalog: actuators, batteries, sensors, structural
stock. This is what turns a grammar terminal from an abstract parameter
(seg_rad=0.02, seg_mass=0.15) into a real, buildable part with a torque-speed
curve, a thermal limit, a cost, and a lead time.

The numbers below are representative off-the-shelf values (the kind printed
on a vendor datasheet or spec sheet for parts in this size class -- smart
servos like Dynamixel/MyActuator, structural aluminum extrusion, standard
LiPo packs), not a verified pull from a specific vendor's current catalog.
Swap in exact datasheet numbers for whichever parts you actually buy; the
schema (what fields a part needs) is the part of this file that matters most
for the rest of the pipeline, not these specific placeholder values.

Units throughout: SI (N, m, kg, W, rad/s, N*m), cost in USD, lead_time_days
in calendar days.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math


# --------------------------------------------------------------------------
# Actuators
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ActuatorSpec:
    part_id: str
    mass: float                 # kg
    gear_ratio: float
    stall_torque: float         # N*m, at the output shaft, post-gearbox
    no_load_speed: float        # rad/s, at the output shaft, post-gearbox
    rated_torque: float         # N*m, continuous (thermally sustainable)
    rated_current: float        # A, continuous
    peak_current: float         # A, short-duration (a few seconds)
    voltage: float               # V, nominal bus voltage
    thermal_time_constant: float  # s, time to reach ~63% of steady-state temp rise at rated current
    max_winding_temp: float      # deg C, winding limit
    ambient_temp: float = 25.0   # deg C, assumed ambient
    thermal_resistance: float = 4.0  # deg C / W, case-to-ambient
    cost: float = 0.0            # USD
    lead_time_days: int = 1
    housing_radius: float = 0.020  # m, approximate cylindrical housing envelope
    housing_length: float = 0.040  # m, approximate housing length along output shaft axis

    def torque_at_speed(self, speed: float) -> float:
        """Linear torque-speed curve (DC-motor approximation): torque falls
        off linearly from stall_torque at speed=0 to 0 at no_load_speed.
        Clipped to stall_torque for speed<=0 and to 0 beyond no_load_speed."""
        if self.no_load_speed <= 0:
            return self.stall_torque
        frac = max(0.0, 1.0 - abs(speed) / self.no_load_speed)
        return self.stall_torque * frac

    def continuous_power(self) -> float:
        """Rated continuous electrical power draw, I^2-limited."""
        return self.voltage * self.rated_current

    def steady_state_temp_rise(self, current: float) -> float:
        """Steady-state winding temperature rise above ambient at a given
        continuous current, via a simple I^2*R-equivalent thermal-resistance
        model: delta_T = thermal_resistance * I^2 * (rated_current-equivalent
        resistive loss). We normalize so that at current=rated_current the
        rise is calibrated to leave a small headroom below max_winding_temp,
        which is how these ratings are set on a real datasheet."""
        scale = (self.max_winding_temp - self.ambient_temp) * 0.85
        if self.rated_current <= 0:
            return 0.0
        return scale * (current / self.rated_current) ** 2

    def thermally_feasible(self, current: float) -> bool:
        return self.ambient_temp + self.steady_state_temp_rise(current) <= self.max_winding_temp


ACTUATORS: Dict[str, ActuatorSpec] = {
    # Small smart-servo class (e.g. Dynamixel XL430 / XC430 size class):
    # light limbs, feet, low-torque joints.
    "servo_small": ActuatorSpec(
        part_id="servo_small", mass=0.072, gear_ratio=258.0,
        stall_torque=1.4, no_load_speed=5.8, rated_torque=0.6,
        rated_current=0.7, peak_current=1.5, voltage=12.0,
        thermal_time_constant=90.0, max_winding_temp=80.0,
        cost=50.0, lead_time_days=3, housing_radius=0.016, housing_length=0.035,
    ),
    # Mid-size smart-servo class (e.g. Dynamixel XM430-W350 size class):
    # default choice for a quadruped/hexapod leg hinge.
    "servo_medium": ActuatorSpec(
        part_id="servo_medium", mass=0.082, gear_ratio=353.0,
        stall_torque=4.1, no_load_speed=4.6, rated_torque=1.8,
        rated_current=1.3, peak_current=2.9, voltage=12.0,
        thermal_time_constant=120.0, max_winding_temp=80.0,
        cost=90.0, lead_time_days=3, housing_radius=0.023, housing_length=0.047,
    ),
    # Larger geared BLDC class (e.g. MyActuator RMD-X / quasi-direct-drive
    # size class): torso-adjacent joints, heavier payload limbs.
    "servo_large": ActuatorSpec(
        part_id="servo_large", mass=0.34, gear_ratio=36.0,
        stall_torque=9.0, no_load_speed=21.0, rated_torque=3.0,
        rated_current=5.0, peak_current=15.0, voltage=24.0,
        thermal_time_constant=200.0, max_winding_temp=100.0,
        cost=220.0, lead_time_days=7, housing_radius=0.029, housing_length=0.040,
    ),
    # Quasi-direct-drive class for whole-body-balance / parkour-grade
    # joints that need high peak torque for impact absorption.
    "qdd_balance": ActuatorSpec(
        part_id="qdd_balance", mass=0.55, gear_ratio=6.0,
        stall_torque=24.0, no_load_speed=33.0, rated_torque=9.0,
        rated_current=18.0, peak_current=60.0, voltage=48.0,
        thermal_time_constant=240.0, max_winding_temp=120.0,
        cost=650.0, lead_time_days=21, housing_radius=0.040, housing_length=0.050,
    ),
}


# --------------------------------------------------------------------------
# Batteries
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class BatterySpec:
    part_id: str
    mass: float          # kg
    capacity_wh: float    # Wh
    nominal_voltage: float  # V
    max_continuous_discharge_a: float  # A
    cost: float = 0.0
    lead_time_days: int = 2

    def endurance_hours(self, avg_power_w: float) -> float:
        if avg_power_w <= 0:
            return float("inf")
        return self.capacity_wh / avg_power_w


BATTERIES: Dict[str, BatterySpec] = {
    "lipo_3s_1500": BatterySpec(
        part_id="lipo_3s_1500", mass=0.13, capacity_wh=16.65,
        nominal_voltage=11.1, max_continuous_discharge_a=30.0,
        cost=18.0, lead_time_days=2,
    ),
    "lipo_4s_4000": BatterySpec(
        part_id="lipo_4s_4000", mass=0.42, capacity_wh=59.2,
        nominal_voltage=14.8, max_continuous_discharge_a=80.0,
        cost=42.0, lead_time_days=2,
    ),
    "li_ion_6s_8000": BatterySpec(
        part_id="li_ion_6s_8000", mass=0.95, capacity_wh=177.6,
        nominal_voltage=22.2, max_continuous_discharge_a=40.0,
        cost=140.0, lead_time_days=5,
    ),
}


# --------------------------------------------------------------------------
# Sensors
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SensorSpec:
    part_id: str
    mass: float          # kg
    power_w: float        # W, continuous draw
    update_rate_hz: float
    cost: float = 0.0
    lead_time_days: int = 1
    kind: str = "generic"  # imu | depth_camera | rgb_camera | force_torque | encoder


SENSORS: Dict[str, SensorSpec] = {
    "imu_6dof": SensorSpec("imu_6dof", mass=0.004, power_w=0.1, update_rate_hz=1000.0,
                           cost=15.0, lead_time_days=1, kind="imu"),
    "depth_cam_rs_d435": SensorSpec("depth_cam_rs_d435", mass=0.072, power_w=1.8,
                                    update_rate_hz=30.0, cost=220.0, lead_time_days=5,
                                    kind="depth_camera"),
    "rgb_cam_global_shutter": SensorSpec("rgb_cam_global_shutter", mass=0.018, power_w=0.9,
                                         update_rate_hz=60.0, cost=95.0, lead_time_days=5,
                                         kind="rgb_camera"),
    "ft_sensor_wrist": SensorSpec("ft_sensor_wrist", mass=0.085, power_w=0.5,
                                  update_rate_hz=1000.0, cost=850.0, lead_time_days=14,
                                  kind="force_torque"),
    "joint_encoder": SensorSpec("joint_encoder", mass=0.006, power_w=0.05,
                                update_rate_hz=1000.0, cost=12.0, lead_time_days=1,
                                kind="encoder"),
}


# --------------------------------------------------------------------------
# Structural stock (round tube/extrusion used for segments)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class StructuralStock:
    part_id: str
    material: str
    density: float        # kg/m^3
    yield_strength: float  # Pa
    elastic_modulus: float  # Pa
    outer_radius: float    # m
    wall_thickness: float  # m
    cost_per_m: float = 0.0
    lead_time_days: int = 3

    @property
    def inner_radius(self) -> float:
        return max(0.0, self.outer_radius - self.wall_thickness)

    def cross_section_area(self) -> float:
        return math.pi * (self.outer_radius ** 2 - self.inner_radius ** 2)

    def second_moment_of_area(self) -> float:
        """Second moment of area I for a hollow circular tube (bending)."""
        return (math.pi / 4.0) * (self.outer_radius ** 4 - self.inner_radius ** 4)

    def mass_per_m(self) -> float:
        return self.cross_section_area() * self.density

    def section_modulus(self) -> float:
        return self.second_moment_of_area() / self.outer_radius


STRUCTURAL_STOCK: Dict[str, StructuralStock] = {
    # 6061-T6 aluminum round tube, thin wall -- light limbs.
    "al6061_tube_light": StructuralStock(
        "al6061_tube_light", material="Al 6061-T6", density=2700.0,
        yield_strength=241e6, elastic_modulus=68.9e9,
        outer_radius=0.010, wall_thickness=0.0015,
        cost_per_m=6.5, lead_time_days=3,
    ),
    # 6061-T6 aluminum round tube, thicker wall -- torso-adjacent / payload
    # limbs.
    "al6061_tube_heavy": StructuralStock(
        "al6061_tube_heavy", material="Al 6061-T6", density=2700.0,
        yield_strength=241e6, elastic_modulus=68.9e9,
        outer_radius=0.018, wall_thickness=0.003,
        cost_per_m=14.0, lead_time_days=3,
    ),
    # Carbon-fiber tube -- high stiffness-to-weight, for parkour/impact
    # limbs; brittle failure mode so we keep the same yield-strength-style
    # field as a conservative first-failure stress rather than ultimate.
    "cf_tube": StructuralStock(
        "cf_tube", material="Carbon fiber (unidirectional)", density=1600.0,
        yield_strength=600e6, elastic_modulus=135e9,
        outer_radius=0.012, wall_thickness=0.001,
        cost_per_m=38.0, lead_time_days=10,
    ),
    # PLA/PETG 3D-printed stock -- cheap, fast, for prototyping /
    # non-load-bearing or lightly loaded segments and feet.
    "printed_petg": StructuralStock(
        "printed_petg", material="PETG (FDM, 100% infill equivalent)",
        density=1270.0, yield_strength=50e6, elastic_modulus=2.1e9,
        outer_radius=0.014, wall_thickness=0.004,
        cost_per_m=2.0, lead_time_days=1,
    ),
}


# Standard off-the-shelf outer diameters for round tube/extrusion stock, by
# material family -- a segment radius that doesn't land near one of these
# needs a custom-drawn or custom-machined part (extra cost, extra lead
# time) rather than an off-the-shelf cut-to-length piece. Values in meters
# (metric tube catalog sizes, common small-diameter sizes only).
STANDARD_OD_BY_MATERIAL: Dict[str, List[float]] = {
    "Al 6061-T6": [0.006, 0.008, 0.010, 0.012, 0.016, 0.018, 0.020, 0.025,
                  0.030, 0.035, 0.040, 0.050],
    "Carbon fiber (unidirectional)": [0.006, 0.008, 0.010, 0.012, 0.016,
                                      0.020, 0.025, 0.030],
    # FDM-printed parts aren't drawn to a fixed OD at all -- any radius is
    # "standard" in the sense that no custom tooling is needed, only print
    # time/material, so there is no OD list for this material.
}


def default_actuator_for(role: str) -> str:
    """Grammar-facing convenience: pick a default catalog part id for a
    given structural role, so existing plans that don't specify a part
    explicitly still get a *real* part rather than an abstract parameter."""
    return {
        "limb_hinge": "servo_medium",
        "torso_adjacent": "servo_large",
        "distal": "servo_small",
        "balance_critical": "qdd_balance",
    }.get(role, "servo_medium")


def bom_line_cost(part_id: str, count: int, registries=(ACTUATORS, BATTERIES, SENSORS,
                                                         STRUCTURAL_STOCK)) -> Tuple[float, int]:
    """Look up a part across every registry and return (unit_cost, lead_time_days)."""
    for reg in registries:
        if part_id in reg:
            spec = reg[part_id]
            cost = getattr(spec, "cost", None)
            if cost is None:
                cost = getattr(spec, "cost_per_m", 0.0)
            return float(cost), int(getattr(spec, "lead_time_days", 1))
    return 0.0, 0
