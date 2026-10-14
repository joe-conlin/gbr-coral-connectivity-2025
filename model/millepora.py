"""Vertical-transport rules used by the final coral dispersal model.

OpenDrift supplies horizontal movement from ROMS. This module adds the early
surface phase and the EVD, EVD-SRC, EVD-LR, SURF and EVDW rules described in
the report. OpenDrift uses z=0 at the surface and negative values below it.
"""

import numpy as np
from opendrift.models.oceandrift import Lagrangian3DArray, OceanDrift


def surface_velocity(age_seconds, settings):
    """Apply buoyant ascent for eight hours, then no imposed movement."""
    return np.where(
        np.asarray(age_seconds) < settings["surface_phase_hours"] * 3600,
        settings["buoyancy_velocity_m_s"],
        0.0,
    )


def surface_bounded_position(z, velocity, seconds):
    """Apply a vertical velocity without allowing z above the surface."""
    return np.minimum(0.0, np.asarray(z) + np.asarray(velocity) * seconds)


def age_limited_windage(age_seconds, settings):
    """Use the EVDW wind factor only during the eight-hour surface phase."""
    active = np.asarray(age_seconds) < settings["maximum_age_hours"] * 3600
    return np.where(active, settings["wind_drift_factor"], 0.0)


def empirical_target_fraction(age_seconds, vertical_rank, settings):
    """Map each fixed particle rank to an age-dependent depth fraction."""
    age_hours = np.asarray(age_seconds) / 3600
    ranks = np.asarray(vertical_rank)
    anchors = settings["distribution_anchors"]
    anchor_hours = np.asarray([item["age_hours"] for item in anchors])
    probabilities = np.asarray([item["probabilities"] for item in anchors])
    fractions = np.linspace(
        settings["minimum_depth_fraction"],
        settings["maximum_depth_fraction"],
        probabilities.shape[1],
    )

    targets = np.empty((len(anchors), len(ranks)))
    for row, probability in enumerate(probabilities):
        bins = np.searchsorted(np.cumsum(probability), ranks, side="right")
        targets[row] = fractions[np.clip(bins, 0, len(fractions) - 1)]

    lower = np.clip(
        np.searchsorted(anchor_hours, age_hours, side="right") - 1,
        0,
        len(anchors) - 1,
    )
    upper = np.minimum(lower + 1, len(anchors) - 1)
    span = anchor_hours[upper] - anchor_hours[lower]
    weight = np.divide(
        age_hours - anchor_hours[lower],
        span,
        out=np.zeros_like(age_hours, dtype=float),
        where=span > 0,
    )
    weight = np.clip(weight, 0, 1)
    particle = np.arange(len(ranks))
    return (
        (1 - weight) * targets[lower, particle]
        + weight * targets[upper, particle]
    )


def target_depth(water_depth, target_fraction, limit_to_source, source_depth):
    """Convert a relative EVD target into metres below the sea surface."""
    available = np.maximum(0.0, np.asarray(water_depth))
    if limit_to_source:
        available = np.minimum(available, np.maximum(0.0, np.asarray(source_depth)))
    return available * np.asarray(target_fraction)


def empirical_velocity(
    z,
    water_depth,
    source_depth,
    age_seconds,
    vertical_rank,
    settings,
    time_step_seconds,
):
    """Move an EVD particle toward its target, subject to the speed cap."""
    fraction = empirical_target_fraction(age_seconds, vertical_rank, settings)
    target_z = -target_depth(
        water_depth,
        fraction,
        settings["limit_to_source_depth"],
        source_depth,
    )
    required = (target_z - np.asarray(z)) / time_step_seconds
    movement = np.clip(
        required,
        -settings["maximum_vertical_speed_m_s"],
        settings["maximum_vertical_speed_m_s"],
    )
    return np.where(
        np.asarray(age_seconds) < settings["surface_phase_hours"] * 3600,
        settings["buoyancy_velocity_m_s"],
        movement,
    )


def smooth_light_weight(solar_elevation, twilight_scale_degrees):
    """Map solar elevation smoothly from darkness (0) to daylight (1)."""
    return 0.5 * (
        1 + np.tanh(np.asarray(solar_elevation) / twilight_scale_degrees)
    )


def astronomical_day_fraction(timestamp, latitude_degrees):
    """Approximate the local fraction of a day spent in daylight."""
    day = timestamp.timetuple().tm_yday
    hour = timestamp.hour + timestamp.minute / 60 + timestamp.second / 3600
    gamma = 2 * np.pi / 365 * (day - 1 + (hour - 12) / 24)
    declination = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )
    latitude = np.deg2rad(np.asarray(latitude_degrees))
    sunset_angle = np.arccos(
        np.clip(-np.tan(latitude) * np.tan(declination), -1, 1)
    )
    return sunset_angle / np.pi


def response_multiplier_from_rank(response_rank, settings):
    """Return the bounded light-response multiplier assigned at release."""
    return np.interp(
        np.asarray(response_rank),
        settings["response_multiplier_quantiles"],
        settings["response_multiplier_values"],
    )


def light_response_velocity(
    z,
    water_depth,
    source_depth,
    age_seconds,
    vertical_rank,
    response_multiplier,
    solar_elevation,
    day_fraction,
    settings,
):
    """Combine the soft EVD target and centred EVD-LR light response."""
    fraction = empirical_target_fraction(age_seconds, vertical_rank, settings)
    target_z = -target_depth(water_depth, fraction, True, source_depth)
    restoring = (target_z - np.asarray(z)) / (
        settings["restoring_timescale_hours"] * 3600
    )
    light = smooth_light_weight(
        solar_elevation,
        settings["twilight_scale_degrees"],
    )
    contrast = (
        settings["daylight_velocity_m_s"]
        - settings["darkness_velocity_m_s"]
    )
    response = (
        contrast
        * (light - np.asarray(day_fraction))
        * np.asarray(response_multiplier)
    )
    movement = np.clip(
        restoring + response,
        -settings["maximum_vertical_speed_m_s"],
        settings["maximum_vertical_speed_m_s"],
    )
    return np.where(
        np.asarray(age_seconds) < settings["surface_phase_hours"] * 3600,
        settings["buoyancy_velocity_m_s"],
        movement,
    )


class SurfaceBoundedDrift(OceanDrift):
    """Apply terminal velocity while enforcing surface and seafloor bounds."""

    def vertical_buoyancy(self):
        self.elements.z[:] = surface_bounded_position(
            self.elements.z,
            self.elements.terminal_velocity,
            self.time_step.total_seconds(),
        )
        seafloor = -(
            self.environment.sea_floor_depth_below_sea_level
            + self.environment.sea_surface_height
        )
        if np.any(self.elements.z < seafloor):
            self.interact_with_seafloor()
            self.bottom_interaction(seafloor)


class SurfaceMilleporaDrift(SurfaceBoundedDrift):
    """SURF comparison: eight-hour buoyancy and no later vertical motion."""

    def __init__(self, behaviour, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.behaviour = behaviour

    def update_terminal_velocity(self, Tprofiles=None, Sprofiles=None, z_index=None):
        self.elements.terminal_velocity[:] = surface_velocity(
            self.elements.age_seconds,
            self.behaviour,
        )


class EmpiricalMilleporaElement(Lagrangian3DArray):
    """Store the fixed EVD ranks and local/source water depths."""

    variables = Lagrangian3DArray.add_variables([
        ("vertical_rank", {
            "dtype": np.float32,
            "units": "1",
            "description": "Fixed quantile rank in the EVD distribution",
            "default": 0.5,
        }),
        ("local_water_depth", {
            "dtype": np.float32,
            "units": "m",
            "description": "Current sea-surface-to-seafloor distance",
            "default": 0,
        }),
        ("source_water_depth", {
            "dtype": np.float32,
            "units": "m",
            "description": "Water depth at the release location",
            "default": 0,
        }),
        ("response_multiplier", {
            "dtype": np.float32,
            "units": "1",
            "description": "Fixed EVD-LR response multiplier",
            "default": 1,
        }),
    ])


class EmpiricalMilleporaDrift(SurfaceBoundedDrift):
    """EVD, EVD-SRC and EVDW model."""

    ElementType = EmpiricalMilleporaElement

    def __init__(self, behaviour, windage=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.behaviour = behaviour
        self.windage = windage

    def update(self):
        if self.windage is not None:
            self.elements.wind_drift_factor[:] = age_limited_windage(
                self.elements.age_seconds,
                self.windage,
            )
        super().update()

    def valid_water_depth(self):
        """Use current ROMS depth, falling back to the last valid value."""
        depth = np.asarray(
            self.environment.sea_floor_depth_below_sea_level
            + self.environment.sea_surface_height
        )
        valid = (
            np.isfinite(depth)
            & (depth > 0)
            & (depth < self.behaviour["invalid_water_depth_threshold_m"])
        )
        previous = np.asarray(self.elements.local_water_depth)
        previous_valid = (
            np.isfinite(previous)
            & (previous > 0)
            & (previous < self.behaviour["invalid_water_depth_threshold_m"])
        )
        depth = np.where(valid, depth, previous)
        if np.any(~valid & ~previous_valid):
            raise RuntimeError("No valid current or previous ROMS water depth")

        missing_source = self.elements.source_water_depth <= 0
        self.elements.source_water_depth[missing_source] = depth[missing_source]
        self.elements.local_water_depth[:] = depth
        self.environment.sea_floor_depth_below_sea_level[:] = (
            depth - self.environment.sea_surface_height
        )
        return depth

    def update_terminal_velocity(self, Tprofiles=None, Sprofiles=None, z_index=None):
        depth = self.valid_water_depth()
        self.elements.terminal_velocity[:] = empirical_velocity(
            self.elements.z,
            depth,
            self.elements.source_water_depth,
            self.elements.age_seconds,
            self.elements.vertical_rank,
            self.behaviour,
            self.time_step.total_seconds(),
        )


class LightResponseMilleporaDrift(EmpiricalMilleporaDrift):
    """EVD-LR soft target, light response and constant vertical mixing."""

    def update_terminal_velocity(self, Tprofiles=None, Sprofiles=None, z_index=None):
        depth = self.valid_water_depth()
        self.elements.terminal_velocity[:] = light_response_velocity(
            z=self.elements.z,
            water_depth=depth,
            source_depth=self.elements.source_water_depth,
            age_seconds=self.elements.age_seconds,
            vertical_rank=self.elements.vertical_rank,
            response_multiplier=self.elements.response_multiplier,
            solar_elevation=self.solar_elevation(),
            day_fraction=astronomical_day_fraction(self.time, self.elements.lat),
            settings=self.behaviour,
        )
