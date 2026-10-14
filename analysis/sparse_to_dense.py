"""Convert a sparse connectivity edge list to a dense CSV matrix."""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("edge_file", type=Path)
    parser.add_argument("output_file", type=Path)
    parser.add_argument(
        "--value",
        choices=["connectivity", "connectivity_mortality_weighted"],
        default="connectivity_mortality_weighted",
    )
    args = parser.parse_args()
    edges = pd.read_csv(
        args.edge_file,
        dtype={"source_gbr_id": str, "destination_gbr_id": str},
    )
    matrix = edges.pivot(
        index="source_gbr_id",
        columns="destination_gbr_id",
        values=args.value,
    ).fillna(0)
    matrix.to_csv(args.output_file)
    print(f"Wrote {matrix.shape[0]} x {matrix.shape[1]} matrix to {args.output_file}")


if __name__ == "__main__":
    main()
