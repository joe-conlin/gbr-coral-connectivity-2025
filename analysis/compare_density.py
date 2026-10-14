"""Compare successive EVD density cases using the report metrics.

Each input must be a completed run directory containing ``analysis``.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def density(run):
    config = yaml.safe_load((run / "run_config.yaml").read_text())
    return float(config["selected_density_per_km2_per_bin"])


def read_edges(run, product):
    path = run / "analysis" / product / "connectivity_edges.csv"
    frame = pd.read_csv(
        path,
        dtype={"source_gbr_id": str, "destination_gbr_id": str},
    )
    return frame.set_index(["source_gbr_id", "destination_gbr_id"])


def compare(left, right):
    ids = left.index.union(right.index)
    shared = left.index.intersection(right.index)
    raw_left = left.reindex(ids, fill_value=0).connectivity
    raw_right = right.reindex(ids, fill_value=0).connectivity
    weighted_left = left.reindex(ids, fill_value=0).connectivity_mortality_weighted
    weighted_right = right.reindex(ids, fill_value=0).connectivity_mortality_weighted
    left_destinations = left.groupby(level="destination_gbr_id").connectivity.sum()
    right_destinations = right.groupby(level="destination_gbr_id").connectivity.sum()
    return {
        "edge_jaccard": len(shared) / len(ids),
        "raw_bray_curtis": 1
        - np.abs(raw_left - raw_right).sum() / (raw_left.sum() + raw_right.sum()),
        "mortality_weighted_bray_curtis": 1
        - np.abs(weighted_left - weighted_right).sum()
        / (weighted_left.sum() + weighted_right.sum()),
        "top10_destination_overlap": len(
            set(left_destinations.nlargest(10).index)
            & set(right_destinations.nlargest(10).index)
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("density_comparison.csv"))
    args = parser.parse_args()
    runs = sorted((path.resolve() for path in args.runs), key=density)

    rows = []
    for product in ("POT", "FIRST"):
        for left_run, right_run in zip(runs[:-1], runs[1:]):
            rows.append({
                "product": product,
                "left_density": density(left_run),
                "right_density": density(right_run),
                **compare(
                    read_edges(left_run, product),
                    read_edges(right_run, product),
                ),
            })
    result = pd.DataFrame(rows)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
