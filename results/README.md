# Results guide

## Connectivity matrices

The final connectivity matrices are stored as sparse edge lists:

- `matrices/density/`: reference EVD at D005, D020, D040, D080 and D120;
- `matrices/transport/`: D080 EVD-SRC, EVD-LR and SURF cases;
- `matrices/windage/`: D040 EVDW case.

The matched reference files are not duplicated. Use
`density/EVD_D080_*.csv` when comparing the D080 transport cases and
`density/EVD_D040_*.csv` when comparing EVDW.

Each matrix file contains:

| Column | Meaning |
|---|---|
| `source_gbr_id` | Source reef identifier |
| `destination_gbr_id` | Destination reef identifier |
| `event_count` | Unweighted contacts for the edge |
| `mortality_weighted_count` | Sum of expected survival weights |
| `particles_released` | Particles released from that source reef |
| `connectivity` | `event_count / particles_released` |
| `connectivity_mortality_weighted` | `mortality_weighted_count / particles_released` |

Absent source-destination pairs have zero connectivity. `POT` retains every
unique competent particle-destination encounter. `FIRST` retains only the
first competent reef contact for each particle.

Use `../analysis/sparse_to_dense.py` if a conventional dense matrix is needed.

## Supporting files

- `reef_metadata.csv` maps all destination GBR IDs to labels, names and
  representative coordinates.
- `source_reefs_domain_c.csv` identifies the 346 source reefs and release area.
- `summaries/` contains the report-level density, transport, spawning-night,
  windage, self-contact and Infomap values.
- `figures/` contains the final Results and Discussion figures.

## Interpretation

These are horizontal reef-polygon contacts made after modelled competency.
They do not include fine-scale settlement habitat, settlement cues or
post-settlement survival. Mortality is an expected survival weight and does
not remove particles. Individual edge presence and strength remained
sensitive to particle density; the report therefore interprets broad patterns,
leading destinations and aggregate values rather than treating individual
links as confirmed or converged.
