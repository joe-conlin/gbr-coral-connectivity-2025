"""Run one final-report coral trajectory case.

Examples
--------
Reference case::

    python model/run_model.py --profile EVD --density 0.8

Early-windage comparison::

    python model/run_model.py --profile EVDW --density 0.4
"""

import argparse
import copy
import json
from datetime import date, datetime, time, timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import yaml
from opendrift.readers import reader_ROMS_native, reader_netCDF_CF_generic
from pyproj import Transformer

try:
    from .millepora import (
        EmpiricalMilleporaDrift,
        LightResponseMilleporaDrift,
        SurfaceMilleporaDrift,
        response_multiplier_from_rank,
    )
except ImportError:
    from millepora import (
        EmpiricalMilleporaDrift,
        LightResponseMilleporaDrift,
        SurfaceMilleporaDrift,
        response_multiplier_from_rank,
    )


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONFIG_PATH = HERE / "config.yaml"


def load_config():
    return yaml.safe_load(CONFIG_PATH.read_text())


def project_path(relative_path):
    """Resolve repository paths independently of the working directory."""
    return ROOT / relative_path


def release_times(config):
    """Build the 183 inclusive three-minute release bins."""
    release = config["release"]
    start_clock = time.fromisoformat(release["start_utc"])
    end_clock = time.fromisoformat(release["end_utc"])
    interval = timedelta(minutes=release["interval_minutes"])
    result = []
    for day_text in release["dates_utc"]:
        day = date.fromisoformat(str(day_text))
        current = datetime.combine(day, start_clock)
        final = datetime.combine(day, end_clock)
        while current <= final:
            result.append(current)
            current += interval
    return result


def sample_polygon(polygon, number, rng):
    """Uniformly sample projected points within one source polygon."""
    min_x, min_y, max_x, max_y = polygon.bounds
    x_result, y_result = [], []
    while len(x_result) < number:
        batch = max(100, 2 * (number - len(x_result)))
        x = rng.uniform(min_x, max_x, batch)
        y = rng.uniform(min_y, max_y, batch)
        inside = shapely.contains_xy(polygon, x, y)
        x_result.extend(x[inside])
        y_result.extend(y[inside])
    return np.asarray(x_result[:number]), np.asarray(y_result[:number])


def make_release_points(config, density):
    """Allocate particles by clipped reef area at every release bin."""
    release = config["release"]
    rng = np.random.default_rng(config["project"]["release_seed"])
    times = release_times(config)
    sources = gpd.read_file(
        project_path(config["paths"]["spatial_inputs"]),
        layer="sources",
    )
    sources = sources.loc[sources.domain.eq(release["domain"])].to_crs("EPSG:3577")
    to_lonlat = Transformer.from_crs("EPSG:3577", "EPSG:4326", always_xy=True)

    records = []
    for source in sources.itertuples():
        per_bin = max(1, round(source.release_area_km2 * density))
        total = per_bin * len(times)
        x, y = sample_polygon(source.geometry, total, rng)
        lon, lat = to_lonlat.transform(x, y)
        records.extend(
            zip(
                np.repeat(source.source_index, total),
                np.repeat(source.GBR_ID, total),
                lon,
                lat,
                np.repeat(times, per_bin),
            )
        )

    columns = ["source_index", "source_gbr_id", "lon", "lat", "release_time_utc"]
    points = pd.DataFrame(records, columns=columns).sort_values(
        ["release_time_utc", "source_index"],
        ignore_index=True,
    )
    points.insert(0, "particle_id", np.arange(len(points), dtype=int))
    return points


def add_particle_ranks(points, config, profile):
    """Assign fixed EVD and, where needed, light-response ranks."""
    if profile == "SURF":
        return points

    ranks = (np.arange(len(points)) + 0.5) / len(points)
    np.random.default_rng(config["evd"]["assignment_seed"]).shuffle(ranks)
    points["vertical_rank"] = ranks

    if profile == "EVD-LR":
        response_ranks = (np.arange(len(points)) + 0.5) / len(points)
        light = config["light_response"]
        np.random.default_rng(light["assignment_seed"]).shuffle(response_ranks)
        points["response_multiplier"] = response_multiplier_from_rank(
            response_ranks,
            light,
        )
    return points


def roms_files(config, start, end):
    """Return the consecutive daily ROMS files needed by the run."""
    directory = project_path(config["paths"]["roms_directory"])
    files = []
    day = start.date()
    while day <= end.date():
        path = directory / f"gbr_his_{day:%Y%m%d}.nc"
        if not path.exists():
            raise FileNotFoundError(f"Missing ROMS forcing: {path}")
        files.append(str(path))
        day += timedelta(days=1)
    return files


def behaviour_settings(config, profile):
    """Combine shared settings with the selected final-report profile."""
    settings = {
        **copy.deepcopy(config["surface_phase"]),
        **copy.deepcopy(config["evd"]),
        **copy.deepcopy(config["profiles"][profile]),
    }
    settings["surface_phase_hours"] = settings.pop("duration_hours")
    settings["limit_to_source_depth"] = settings.pop("source_depth_limit")
    if profile == "EVD-LR":
        settings.update(copy.deepcopy(config["light_response"]))
    return settings


def configure_model(config, profile, readers, logfile):
    """Create OpenDrift and explicitly set the enabled numerical processes."""
    numerics = config["numerics"]
    profile_settings = config["profiles"][profile]
    behaviour = behaviour_settings(config, profile)

    if profile == "SURF":
        model = SurfaceMilleporaDrift(behaviour, loglevel=20, logfile=logfile)
    elif profile == "EVD-LR":
        model = LightResponseMilleporaDrift(behaviour, loglevel=20, logfile=logfile)
    else:
        windage = config["windage"] if profile == "EVDW" else None
        model = EmpiricalMilleporaDrift(
            behaviour,
            windage=windage,
            loglevel=20,
            logfile=logfile,
        )

    model.add_reader(readers)
    model.set_config("general:use_auto_landmask", numerics["use_auto_landmask"])
    model.set_config("general:coastline_action", numerics["coastline_action"])
    model.set_config("general:seafloor_action", numerics["seafloor_action"])
    model.set_config("drift:advection_scheme", numerics["advection_scheme"])
    model.set_config("drift:max_speed", numerics["reader_buffer_speed_m_s"])
    model.set_config("drift:vertical_advection", numerics["vertical_advection"])
    model.set_config("drift:vertical_mixing", profile_settings["vertical_mixing"])
    model.set_config("drift:vertical_mixing_at_surface", True)
    model.set_config(
        "vertical_mixing:timestep",
        config["light_response"]["mixing_time_step_seconds"],
    )
    if profile_settings["vertical_mixing"]:
        model.set_config("vertical_mixing:diffusivitymodel", "constant")
        model.set_config(
            "environment:fallback:ocean_vertical_diffusivity",
            config["light_response"]["vertical_diffusivity_m2_s"],
        )
    model.set_config("drift:stokes_drift", numerics["stokes_drift"])
    model.set_config("drift:current_uncertainty", numerics["current_uncertainty_m_s"])
    model.set_config("drift:wind_uncertainty", numerics["wind_uncertainty_m_s"])
    model.set_config(
        "environment:fallback:horizontal_diffusivity",
        numerics["horizontal_diffusivity_m2_s"],
    )
    model.set_config(
        "drift:wind_drift_depth",
        config["windage"]["maximum_depth_m"]
        if profile_settings["direct_windage"]
        else 0,
    )
    model.set_config("seed:wind_drift_factor", 0)
    model.set_config(
        "drift:max_age_seconds",
        numerics["duration_days"] * 86400 + numerics["output_interval_seconds"],
    )
    return model


def output_directory(config, profile, density, override):
    if override:
        return Path(override).resolve()
    density_code = round(density * 100)
    return project_path(config["paths"]["output_root"]) / f"{profile}_D{density_code:03d}"


def main():
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=config["profiles"], default="EVD")
    parser.add_argument(
        "--density",
        type=float,
        default=config["release"]["default_density_per_km2_per_bin"],
        help="Particles per km2 per three-minute release bin",
    )
    parser.add_argument("--output-dir", help="Optional non-existing output directory")
    args = parser.parse_args()
    if args.density <= 0:
        raise ValueError("Density must be positive")

    spatial_inputs = project_path(config["paths"]["spatial_inputs"])
    if not spatial_inputs.exists():
        raise FileNotFoundError(f"Missing processed spatial input: {spatial_inputs}")

    points = make_release_points(config, args.density)
    points = add_particle_ranks(points, config, args.profile)
    duration = timedelta(days=config["numerics"]["duration_days"])
    end_time = points.release_time_utc.max() + duration

    output = output_directory(config, args.profile, args.density, args.output_dir)
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    output.mkdir(parents=True)

    readers = [reader_ROMS_native.Reader(
        roms_files(config, points.release_time_utc.min(), end_time)
    )]
    if config["profiles"][args.profile]["direct_windage"]:
        readers.append(reader_netCDF_CF_generic.Reader(
            str(project_path(config["paths"]["wind_file"])),
            standard_name_mapping={"Uwind": "x_wind", "Vwind": "y_wind"},
        ))

    model = configure_model(
        config,
        args.profile,
        readers,
        logfile=output / "opendrift.log",
    )
    for release_time, group in points.groupby("release_time_utc", sort=True):
        state = {}
        if "vertical_rank" in group:
            state["vertical_rank"] = group.vertical_rank.to_numpy()
        if "response_multiplier" in group:
            state["response_multiplier"] = group.response_multiplier.to_numpy()
        model.seed_elements(
            lon=group.lon.to_numpy(),
            lat=group.lat.to_numpy(),
            z=config["release"]["depth_m"],
            time=release_time.to_pydatetime(),
            terminal_velocity=0,
            wind_drift_factor=(
                config["windage"]["wind_drift_factor"]
                if args.profile == "EVDW"
                else 0
            ),
            origin_marker=group.source_index.to_numpy(dtype=np.int32),
            **state,
        )

    points.to_csv(output / "release_points.csv", index=False)
    resolved = copy.deepcopy(config)
    resolved["selected_profile"] = args.profile
    resolved["selected_density_per_km2_per_bin"] = args.density
    (output / "run_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False)
    )

    export_variables = ["lon", "lat", "z", "status", "age_seconds", "origin_marker"]
    if args.profile != "SURF":
        export_variables.extend([
            "vertical_rank",
            "local_water_depth",
            "source_water_depth",
            "wind_drift_factor",
        ])
    if args.profile == "EVD-LR":
        export_variables.append("response_multiplier")

    model.run(
        duration=end_time - points.release_time_utc.min(),
        time_step=config["numerics"]["time_step_seconds"],
        time_step_output=config["numerics"]["output_interval_seconds"],
        outfile=str(output / "trajectory.nc"),
        export_variables=export_variables,
        export_buffer_length=24,
        stop_on_error=True,
    )
    if model.time < end_time:
        raise RuntimeError(f"Trajectory ended early at {model.time}; see opendrift.log")

    manifest = {
        "profile": args.profile,
        "density_per_km2_per_bin": args.density,
        "particles": len(points),
        "source_reefs": int(points.source_gbr_id.nunique()),
        "first_release_utc": points.release_time_utc.min().isoformat(),
        "last_release_utc": points.release_time_utc.max().isoformat(),
        "requested_end_utc": end_time.isoformat(),
        "actual_end_utc": model.time.isoformat(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
