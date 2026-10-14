"""Fit and compare broad Infomap modules across D040, D080 and D120."""

import argparse
import json
from pathlib import Path

from infomap import Infomap
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score


WEIGHT = "connectivity_mortality_weighted"


def read_network(path):
    edges = pd.read_csv(
        path,
        dtype={"source_gbr_id": str, "destination_gbr_id": str},
    )
    sources = sorted(edges.source_gbr_id.unique())
    internal = edges.loc[
        edges.destination_gbr_id.isin(sources) & edges[WEIGHT].gt(0)
    ].copy()
    retained = internal[WEIGHT].sum() / edges[WEIGHT].sum()
    return sources, internal, float(retained)


def fit(sources, edges, seed=2025, trials=100):
    codes = {reef_id: number for number, reef_id in enumerate(sources)}
    model = Infomap(f"--directed --silent --seed {seed} --num-trials {trials}")
    for edge in edges.itertuples():
        model.add_link(
            codes[edge.source_gbr_id],
            codes[edge.destination_gbr_id],
            getattr(edge, WEIGHT),
        )
    model.run()
    assignment = {
        sources[node.node_id]: int(node.path[0])
        for node in model.tree
        if node.is_leaf
    }
    return assignment, int(model.num_top_modules), float(model.codelength)


def agreement(left, right):
    reef_ids = sorted(set(left) & set(right))
    a = [left[reef_id] for reef_id in reef_ids]
    b = [right[reef_id] for reef_id in reef_ids]
    return {
        "common_reefs": len(reef_ids),
        "adjusted_mutual_information": adjusted_mutual_info_score(a, b),
        "adjusted_rand_index": adjusted_rand_score(a, b),
    }


def align(reference, candidate):
    """Relabel candidate modules to maximise overlap with D080."""
    reference_modules = sorted(set(reference.values()))
    candidate_modules = sorted(set(candidate.values()))
    overlap = np.zeros((len(candidate_modules), len(reference_modules)), dtype=int)
    for reef_id in set(reference) & set(candidate):
        i = candidate_modules.index(candidate[reef_id])
        j = reference_modules.index(reference[reef_id])
        overlap[i, j] += 1
    rows, columns = linear_sum_assignment(-overlap)
    mapping = {
        candidate_modules[row]: reference_modules[column]
        for row, column in zip(rows, columns)
    }
    return {reef_id: mapping.get(module, module) for reef_id, module in candidate.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("d040_first", type=Path)
    parser.add_argument("d080_first", type=Path)
    parser.add_argument("d120_first", type=Path)
    parser.add_argument("--output", type=Path, default=Path("infomap_modules"))
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    output.mkdir(parents=True)

    paths = {"D040": args.d040_first, "D080": args.d080_first, "D120": args.d120_first}
    networks = {label: read_network(path) for label, path in paths.items()}
    fits = {}
    for label, (sources, edges, retained) in networks.items():
        assignment, modules, codelength = fit(sources, edges)
        fits[label] = {
            "assignment": assignment,
            "modules": modules,
            "codelength": codelength,
            "retained_internal_weight": retained,
        }

    reference = fits["D080"]["assignment"]
    aligned = {
        "D040": align(reference, fits["D040"]["assignment"]),
        "D080": reference,
        "D120": align(reference, fits["D120"]["assignment"]),
    }
    module_order = sorted(
        set(reference.values()),
        key=lambda module: (-sum(value == module for value in reference.values()), module),
    )
    display = {module: number + 1 for number, module in enumerate(module_order)}

    assignment_rows = []
    for reef_id, module in reference.items():
        same = (
            aligned["D040"].get(reef_id) == module
            and aligned["D120"].get(reef_id) == module
        )
        assignment_rows.append({
            "gbr_id": reef_id,
            "d040_module_aligned": display.get(aligned["D040"].get(reef_id)),
            "d080_module": display[module],
            "d120_module_aligned": display.get(aligned["D120"].get(reef_id)),
            "same_module_at_all_densities": same,
        })
    pd.DataFrame(assignment_rows).to_csv(output / "assignments.csv", index=False)

    comparisons = []
    for left, right in (("D040", "D080"), ("D080", "D120"), ("D040", "D120")):
        comparisons.append({
            "comparison": f"{left}-{right}",
            **agreement(fits[left]["assignment"], fits[right]["assignment"]),
        })
    pd.DataFrame(comparisons).to_csv(output / "density_agreement.csv", index=False)

    metadata = {
        label: {
            "top_level_modules": values["modules"],
            "codelength": values["codelength"],
            "retained_internal_weight": values["retained_internal_weight"],
        }
        for label, values in fits.items()
    }
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
