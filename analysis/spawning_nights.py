"""Build night-specific POT and FIRST matrices from a pooled EVD run."""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


UTC_OFFSET_HOURS = 10


def matrix_edges(events, releases):
    """Create one sparse, source-normalised connectivity matrix."""
    released = releases.groupby("source_gbr_id").size().rename("particles_released")
    edges = (
        events.groupby(["source_gbr_id", "destination_gbr_id"], as_index=False)
        .agg(
            event_count=("particle_id", "size"),
            mortality_weighted_count=("mortality_weight", "sum"),
        )
        .merge(released, on="source_gbr_id", validate="many_to_one")
    )
    edges["connectivity"] = edges.event_count / edges.particles_released
    edges["connectivity_mortality_weighted"] = (
        edges.mortality_weighted_count / edges.particles_released
    )
    return edges


def edge_series(edges):
    return edges.set_index(
        ["source_gbr_id", "destination_gbr_id"]
    ).connectivity_mortality_weighted


def bray_curtis(left, right):
    ids = left.index.union(right.index)
    a = left.reindex(ids, fill_value=0).to_numpy()
    b = right.reindex(ids, fill_value=0).to_numpy()
    return 1 - np.abs(a - b).sum() / (a.sum() + b.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path, default=Path("spawning_nights"))
    args = parser.parse_args()
    run = args.run_directory.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    output.mkdir(parents=True)

    releases = pd.read_csv(run / "release_points.csv", parse_dates=["release_time_utc"])
    releases["source_gbr_id"] = releases.source_gbr_id.astype(str)
    releases["release_night"] = (
        releases.release_time_utc + pd.Timedelta(hours=UTC_OFFSET_HOURS)
    ).dt.date.astype(str)
    nights = sorted(releases.release_night.unique())

    summaries = []
    comparisons = []
    for product in ("FIRST", "POT"):
        events = pd.read_parquet(run / "analysis" / product / "events.parquet")
        events["source_gbr_id"] = events.source_gbr_id.astype(str)
        events["destination_gbr_id"] = events.destination_gbr_id.astype(str)
        night_lookup = releases.set_index("particle_id").release_night
        events["release_night"] = events.particle_id.map(night_lookup)

        matrices = {}
        destinations = {}
        for night in nights:
            night_releases = releases.loc[releases.release_night.eq(night)]
            night_events = events.loc[events.release_night.eq(night)]
            edges = matrix_edges(night_events, night_releases)
            edges.to_csv(output / f"{night}_{product}.csv", index=False)
            matrices[night] = edge_series(edges)
            destinations[night] = (
                edges.groupby("destination_gbr_id")
                .connectivity_mortality_weighted.sum()
                .sort_values(ascending=False)
            )
            summaries.append({
                "product": product,
                "release_night": night,
                "release_particles": len(night_releases),
                "events": len(night_events),
                "raw_contacts_per_release": len(night_events) / len(night_releases),
                "weighted_contacts_per_release": (
                    night_events.mortality_weight.sum() / len(night_releases)
                ),
                "destination_reefs": night_events.destination_gbr_id.nunique(),
                "nonzero_edges": len(edges),
            })

        for left, right in combinations(nights, 2):
            left_ids = set(matrices[left].index)
            right_ids = set(matrices[right].index)
            comparisons.append({
                "product": product,
                "night_a": left,
                "night_b": right,
                "edge_jaccard": len(left_ids & right_ids) / len(left_ids | right_ids),
                "mortality_weighted_bray_curtis": bray_curtis(
                    matrices[left], matrices[right]
                ),
                "top10_destination_overlap": len(
                    set(destinations[left].head(10).index)
                    & set(destinations[right].head(10).index)
                ),
            })

    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(output / "pairwise_similarity.csv", index=False)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
