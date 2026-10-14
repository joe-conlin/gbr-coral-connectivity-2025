"""Create endpoint, POT and FIRST products from one trajectory run."""

import argparse
import json
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr
import yaml
from shapely.geometry import Point
from shapely.strtree import STRtree


ROOT = Path(__file__).resolve().parent.parent


def write_status(run, stage, completed, total, started):
    (run / "postprocess_status.json").write_text(json.dumps({
        "stage": stage, "completed": completed, "total": total,
        "percent": round(100 * completed / total, 1) if total else 0,
        "elapsed_seconds": round(time.monotonic() - started),
    }) + "\n")


def intersection_position(line, polygon):
    """Return the first fractional position at which line intersects polygon."""
    intersection = line.intersection(polygon)
    if intersection.is_empty:
        # STRtree's broad predicate can rarely disagree with an exact geometry
        # operation. A missing exact intersection is not a reef-contact event.
        return np.nan
    if polygon.covers(Point(line.coords[0])) or line.length == 0:
        return 0.0
    coordinates = shapely.get_coordinates(intersection)
    return min(line.project(Point(xy), normalized=True) for xy in coordinates)


def particle_fates(data, releases, horizon_days):
    """Interpolate every particle position to the exact analysis horizon."""
    times = pd.DatetimeIndex(data.time.values)
    final_lon = np.full(len(releases), np.nan)
    final_lat = np.full(len(releases), np.nan)
    final_status = np.full(len(releases), np.nan)
    endpoint_available = np.zeros(len(releases), dtype=bool)
    groups = releases.groupby("release_time_utc", sort=True).indices
    print(f"PT endpoints: 0/{len(groups)} release bins", flush=True)
    for number, (release_time, particle_ids) in enumerate(groups.items(), start=1):
        target = pd.Timestamp(release_time) + pd.Timedelta(days=horizon_days)
        right = times.searchsorted(target, side="left")
        if right == len(times):
            raise ValueError(f"Release bin {release_time} does not bracket day {horizon_days}")
        left = right if times[right] == target else right - 1
        if left < 0:
            raise ValueError(f"Release bin {release_time} does not bracket day {horizon_days}")
        fraction = (target - times[left]) / (times[right] - times[left]) if left != right else 0.0
        subset = data.isel(trajectory=particle_ids)
        left_lon, right_lon = subset.lon.isel(time=left).values, subset.lon.isel(time=right).values
        left_lat, right_lat = subset.lat.isel(time=left).values, subset.lat.isel(time=right).values
        valid = (
            np.isfinite(left_lon) & np.isfinite(right_lon)
            & np.isfinite(left_lat) & np.isfinite(right_lat)
        )
        valid_ids = particle_ids[valid]
        final_lon[valid_ids] = left_lon[valid] + fraction * (right_lon[valid] - left_lon[valid])
        final_lat[valid_ids] = left_lat[valid] + fraction * (right_lat[valid] - left_lat[valid])
        final_status[valid_ids] = subset.status.isel(time=left).values[valid]
        endpoint_available[valid_ids] = True
        if number == len(groups) or number % 25 == 0:
            print(f"PT endpoints: {number}/{len(groups)} release bins ({number / len(groups):.0%})", flush=True)
    fates = pd.DataFrame({
        "final_lon": final_lon, "final_lat": final_lat,
        "final_age_days": float(horizon_days), "final_status": final_status,
        "endpoint_available": endpoint_available,
    })
    return pd.concat((releases.reset_index(drop=True), fates), axis=1)


def competent_contacts(data, competence_age, reefs, horizon_days, run=None):
    """Return the first competent contact for every particle-destination pair."""
    tree = STRtree(reefs.geometry.to_numpy())
    seen = set()
    events = []

    intervals = data.sizes["time"] - 1
    started = time.monotonic()
    print(f"POT contact scan: 0/{intervals} intervals", flush=True)
    if run is not None:
        write_status(run, "POT contact scan", 0, intervals, started)
    for time_index in range(1, data.sizes["time"]):
        previous_lon = data.lon.isel(time=time_index - 1).values
        previous_lat = data.lat.isel(time=time_index - 1).values
        current_lon = data.lon.isel(time=time_index).values
        current_lat = data.lat.isel(time=time_index).values
        previous_age = data.age_seconds.isel(time=time_index - 1).values / 86400
        current_age = data.age_seconds.isel(time=time_index).values / 86400

        end_fraction = np.clip(
            (horizon_days - previous_age)
            / np.maximum(current_age - previous_age, 1e-12),
            0,
            1,
        )
        current_lon = previous_lon + end_fraction * (current_lon - previous_lon)
        current_lat = previous_lat + end_fraction * (current_lat - previous_lat)
        current_age = np.minimum(current_age, horizon_days)
        eligible = (
            np.isfinite(previous_lon)
            & np.isfinite(previous_lat)
            & np.isfinite(current_lon)
            & np.isfinite(current_lat)
            & np.isfinite(previous_age)
            & np.isfinite(current_age)
            & (previous_age < horizon_days)
            & (current_age >= competence_age)
        )
        particle_ids = np.flatnonzero(eligible)
        if not len(particle_ids):
            continue

        start_fraction = np.clip(
            (competence_age[particle_ids] - previous_age[particle_ids])
            / np.maximum(current_age[particle_ids] - previous_age[particle_ids], 1e-12),
            0,
            1,
        )
        start_lon = previous_lon[particle_ids] + start_fraction * (
            current_lon[particle_ids] - previous_lon[particle_ids]
        )
        start_lat = previous_lat[particle_ids] + start_fraction * (
            current_lat[particle_ids] - previous_lat[particle_ids]
        )
        lines = shapely.linestrings(
            np.stack(
                (
                    np.column_stack((start_lon, start_lat)),
                    np.column_stack((current_lon[particle_ids], current_lat[particle_ids])),
                ),
                axis=1,
            )
        )
        pairs = tree.query(lines, predicate="intersects")
        candidates = []
        for local_line, reef_index in zip(*pairs):
            particle_id = int(particle_ids[local_line])
            destination = reefs.GBR_ID.iloc[reef_index]
            if (particle_id, destination) in seen:
                continue
            line = lines[local_line]
            position = intersection_position(line, reefs.geometry.iloc[reef_index])
            # Do not allow a numerical geometry mismatch to become an infinite
            # age/position event. Valid line positions are bounded fractions.
            if not np.isfinite(position) or not 0 <= position <= 1:
                continue
            age_at_start = previous_age[particle_id] + start_fraction[local_line] * (
                current_age[particle_id] - previous_age[particle_id]
            )
            event_age = age_at_start + position * (
                current_age[particle_id] - age_at_start
            )
            point = line.interpolate(position, normalized=True)
            candidates.append(
                {
                    "particle_id": particle_id,
                    "destination_gbr_id": destination,
                    "competence_age_days": competence_age[particle_id],
                    "encounter_age_days": event_age,
                    "encounter_lon": point.x,
                    "encounter_lat": point.y,
                }
            )
        candidates.sort(
            key=lambda row: (
                row["encounter_age_days"], row["particle_id"], row["destination_gbr_id"]
            )
        )
        for event in candidates:
            key = (event["particle_id"], event["destination_gbr_id"])
            if key not in seen:
                seen.add(key)
                events.append(event)
        if time_index == intervals or time_index % 100 == 0:
            elapsed = time.monotonic() - started
            remaining = elapsed * (intervals / time_index - 1)
            print(
                f"POT contact scan: {time_index}/{intervals} ({time_index / intervals:.0%}); "
                f"elapsed {elapsed / 60:.1f} min, ETA {remaining / 60:.1f} min",
                flush=True,
            )
            if run is not None:
                write_status(run, "POT contact scan", time_index, intervals, started)

    columns = [
        "particle_id", "destination_gbr_id", "competence_age_days",
        "encounter_age_days", "encounter_lon", "encounter_lat",
    ]
    return pd.DataFrame(events, columns=columns).sort_values(
        ["particle_id", "encounter_age_days", "destination_gbr_id"],
        ignore_index=True,
    )


def add_weights(events, releases, mortality_rate):
    events = events.merge(
        releases[["particle_id", "source_gbr_id"]],
        on="particle_id",
        how="left",
        validate="many_to_one",
    )
    events["event_weight"] = 1.0
    events["mortality_weight"] = np.exp(
        -mortality_rate * events.encounter_age_days
    )
    return events


def connectivity_edges(events, releases):
    released = releases.groupby("source_gbr_id").size().rename("particles_released")
    edges = (
        events.groupby(["source_gbr_id", "destination_gbr_id"], as_index=False)
        .agg(event_count=("event_weight", "sum"), mortality_weighted_count=("mortality_weight", "sum"))
        .merge(released, on="source_gbr_id")
    )
    edges["connectivity"] = edges.event_count / edges.particles_released
    edges["connectivity_mortality_weighted"] = (
        edges.mortality_weighted_count / edges.particles_released
    )
    return edges


def write_product(directory, events, releases):
    directory.mkdir()
    events.to_parquet(directory / "events.parquet", index=False)
    connectivity_edges(events, releases).to_csv(
        directory / "connectivity_edges.csv", index=False
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    run = args.run_directory.resolve()
    analysis = run / "analysis"
    if analysis.exists():
        raise FileExistsError(f"Analysis directory already exists: {analysis}")
    analysis.mkdir()

    cfg = yaml.safe_load((run / "run_config.yaml").read_text())
    started = time.monotonic()
    write_status(analysis, "Preparing inputs", 0, 1, started)
    releases = pd.read_csv(run / "release_points.csv", parse_dates=["release_time_utc"])
    reefs = gpd.read_file(ROOT / cfg["paths"]["spatial_inputs"], layer="reefs")
    reefs.GBR_ID = reefs.GBR_ID.astype(str)
    releases.source_gbr_id = releases.source_gbr_id.astype(str)
    biology = cfg["biology"]
    horizon = cfg["numerics"]["duration_days"]
    rng = np.random.default_rng(biology["competency_seed"])
    competence_age = biology["minimum_competency_days"] + rng.exponential(
        1 / biology["competency_rate_per_day"], len(releases)
    )

    with xr.open_dataset(run / "trajectory.nc") as data:
        pt_output = analysis / "PT" / "particle_fates.parquet"
        if pt_output.exists():
            fates = pd.read_parquet(pt_output)
            if (
                len(fates) != len(releases)
                or fates.particle_id.nunique() != len(releases)
                or not np.array_equal(
                    np.sort(fates.particle_id.to_numpy()),
                    np.arange(len(releases)),
                )
            ):
                raise ValueError("Existing PT output does not match the release table")
            print("Reusing verified existing PT endpoints.", flush=True)
        else:
            pt = analysis / ".PT-in-progress"
            pt.mkdir()
            particle_fates(data, releases, horizon).to_parquet(
                pt / "particle_fates.parquet", index=False
            )
            pt.replace(analysis / "PT")
            print("PT endpoints complete.", flush=True)
        potential = competent_contacts(data, competence_age, reefs, horizon, analysis)

    print("POT contact scan complete; writing connectivity products.", flush=True)
    potential = add_weights(potential, releases, biology["mortality_rate_per_day"])
    first = potential.sort_values(
        ["particle_id", "encounter_age_days", "destination_gbr_id"]
    ).drop_duplicates("particle_id")
    write_product(analysis / "POT", potential, releases)
    write_product(analysis / "FIRST", first, releases)

    fates = pd.read_parquet(analysis / "PT/particle_fates.parquet")
    manifest = {
        "source_run_directory": str(run),
        "particles": len(releases),
        "PT": {
            "exact_day_14_endpoints": int(fates.endpoint_available.sum()),
            "missing_data_before_day_14": int((~fates.endpoint_available).sum()),
            "missing_data_treatment": "Retained in release denominators; endpoint left unavailable and contacts evaluated only while valid trajectory data exist.",
        },
        "competent_by_day_14": int(np.sum(competence_age <= horizon)),
        "POT": {
            "unique_particle_destination_encounters": len(potential),
            "particles_with_encounter": int(potential.particle_id.nunique()),
            "destination_reefs_reached": int(potential.destination_gbr_id.nunique()),
            "total_connectivity": len(potential) / len(releases),
            "total_connectivity_mortality_weighted": float(
                potential.mortality_weight.sum() / len(releases)
            ),
        },
        "FIRST": {
            "first_contact_events": len(first),
            "destination_reefs_reached": int(first.destination_gbr_id.nunique()),
            "self_settlements": int(
                (first.source_gbr_id == first.destination_gbr_id).sum()
            ),
            "total_connectivity": len(first) / len(releases),
            "total_connectivity_mortality_weighted": float(
                first.mortality_weight.sum() / len(releases)
            ),
        },
        "competency": "One sampled acquisition age per particle; competency is retained through day 14.",
        "mortality": "Independent deterministic survival weight exp(-0.054 * encounter_age_days).",
    }
    (analysis / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_status(analysis, "Complete", 1, 1, started)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
