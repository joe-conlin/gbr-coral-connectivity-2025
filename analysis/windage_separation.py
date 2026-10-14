"""Measure separation between matched EVD and EVDW trajectories."""

import argparse
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd


AGES_HOURS = np.array([0, 1, 3, 6, 8, 12, 24, 48, 72, 168, 336])
EARTH_RADIUS_KM = 6371.0088
CHUNK_SIZE = 4000


def distance_km(lon_a, lat_a, lon_b, lat_b):
    """Great-circle separation of paired positions."""
    lon_a, lat_a, lon_b, lat_b = map(np.deg2rad, (lon_a, lat_a, lon_b, lat_b))
    dlon = lon_b - lon_a
    dlat = lat_b - lat_a
    haversine = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat_a) * np.cos(lat_b) * np.sin(dlon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(haversine, 0, 1)))


def values(variable, start, stop):
    return np.ma.filled(variable[start:stop], np.nan).astype(np.float32, copy=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evd_trajectory", type=Path)
    parser.add_argument("evdw_trajectory", type=Path)
    parser.add_argument("--output", type=Path, default=Path("windage_separation.csv"))
    args = parser.parse_args()
    separation = {age: [] for age in AGES_HOURS}

    with nc.Dataset(args.evd_trajectory) as evd, nc.Dataset(args.evdw_trajectory) as evdw:
        if not np.array_equal(evd["trajectory"][:], evdw["trajectory"][:]):
            raise ValueError("Trajectory identifiers differ")
        if not np.array_equal(evd["time"][:], evdw["time"][:]):
            raise ValueError("Saved time axes differ")

        total = evd.dimensions["trajectory"].size
        for start in range(0, total, CHUNK_SIZE):
            stop = min(start + CHUNK_SIZE, total)
            age = values(evd["age_seconds"], start, stop)
            lon_a = values(evd["lon"], start, stop)
            lat_a = values(evd["lat"], start, stop)
            lon_b = values(evdw["lon"], start, stop)
            lat_b = values(evdw["lat"], start, stop)
            rows = np.arange(stop - start)

            for age_hours in AGES_HOURS:
                if age_hours == 0:
                    separation[age_hours].append(np.zeros(stop - start))
                    continue
                difference = np.abs(age - age_hours * 3600)
                valid_positions = (
                    np.isfinite(lon_a)
                    & np.isfinite(lat_a)
                    & np.isfinite(lon_b)
                    & np.isfinite(lat_b)
                    & (age > 0)
                )
                difference[~valid_positions] = np.inf
                index = difference.argmin(axis=1)
                found = np.isfinite(difference[rows, index])
                separation[age_hours].append(
                    distance_km(
                        lon_a[rows[found], index[found]],
                        lat_a[rows[found], index[found]],
                        lon_b[rows[found], index[found]],
                        lat_b[rows[found], index[found]],
                    )
                )

    rows = []
    for age_hours in AGES_HOURS:
        sample = np.concatenate(separation[age_hours])
        rows.append({
            "particle_age_h": age_hours,
            "matched_particles": len(sample),
            "median_separation_km": np.median(sample),
            "q05_separation_km": np.quantile(sample, 0.05),
            "q25_separation_km": np.quantile(sample, 0.25),
            "q75_separation_km": np.quantile(sample, 0.75),
            "q95_separation_km": np.quantile(sample, 0.95),
            "mean_separation_km": np.mean(sample),
        })
    result = pd.DataFrame(rows)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
