# GBR coral connectivity 2025

This repository contains the compact model setup and final numerical results
used in the thesis *Spatial Modelling of Acropora millepora Larval Dispersal
and Potential Connectivity in the Central Great Barrier Reef*.

The model estimates **potential connectivity**, not successful settlement or
recruitment. Fertilisation, habitat suitability, settlement cues and
post-settlement survival are outside its scope.

## Repository structure

```text
model/                 trajectory model and one configuration file
analysis/              contact processing and reported comparison methods
inputs/                processed reef, source and domain geometries
results/matrices/      final sparse POT and FIRST connectivity matrices
results/summaries/     numerical summaries used in the report
results/figures/       final Results and Discussion figures
tests/                 focused checks of the movement rules
environment.yml        reproducible Python environment
```

Historical experiments, abandoned formulations, report-building scripts,
animations, notebooks and multi-gigabyte trajectories are intentionally not
included.

## Final model cases

| Profile | Purpose | Reported density |
|---|---|---:|
| `EVD` | Reference empirical vertical distribution | D005, D020, D040, D080 and D120 |
| `EVD-SRC` | EVD constrained by source-water depth | D080 |
| `EVD-LR` | Source-constrained soft EVD target, light response and constant vertical mixing | D080 |
| `SURF` | Surface-current comparison | D080 |
| `EVDW` | EVD with 3% direct windage during the first eight hours | D040 |

`D080` means 0.8 particles per square kilometre of clipped source-reef area
per three-minute release bin. The completed EVD-LR run used the historical
internal label `BPROSOFTKZ`; files in this repository use the report name
`EVD-LR`.

## Required forcing data

The raw forcing is not included because it is approximately 49 GB. Place the
files at the paths defined in `model/config.yaml`:

```text
data/roms/gbr_his_YYYYMMDD.nc
data/gfs/GFS_AUS_20251xxx.nc
```

ROMS supplies hourly three-dimensional horizontal currents, sea-surface
height and bathymetry. GFS wind is read only by the EVDW sensitivity.
`inputs/model_inputs.gpkg` contains the processed spatial input used by the
model: 346 Domain C source reefs and all 3,862 destination reef polygons.

## Environment

```bash
conda env create -f environment.yml
conda activate gbr-connectivity
python -m unittest discover -s tests
```

The report used Python 3.11, OpenDrift 1.14.10 and Infomap 2.15.1.

## Run and post-process one case

Run the D080 EVD reference model from the repository root:

```bash
python model/run_model.py --profile EVD --density 0.8
python analysis/postprocess.py outputs/EVD_D080
```

The first command writes the release table, resolved configuration, trajectory
and run manifest. The second command writes:

```text
outputs/EVD_D080/analysis/PT/particle_fates.parquet
outputs/EVD_D080/analysis/POT/events.parquet
outputs/EVD_D080/analysis/POT/connectivity_edges.csv
outputs/EVD_D080/analysis/FIRST/events.parquet
outputs/EVD_D080/analysis/FIRST/connectivity_edges.csv
```

Production runs are computationally expensive. The completed D080 reference
case contained 513,498 particles and took about 16 hours on the computer
reported in the thesis.

## Analysis scripts

- `postprocess.py` interpolates day-14 endpoints, samples competency ages,
  identifies competent reef-polygon intersections and creates POT/FIRST edges.
- `compare_density.py` calculates edge Jaccard, Bray-Curtis and leading-
  destination overlap between successive density cases.
- `spawning_nights.py` reconstructs separate matrices for 8, 9 and 10 November.
- `infomap_modules.py` compares the broad directed FIRST modules at D040, D080
  and D120.
- `windage_separation.py` measures horizontal separation of matched EVD and
  EVDW particles.
- `sparse_to_dense.py` converts any supplied edge list to a conventional dense
  source-by-destination CSV matrix.

## Results

The final matrices are supplied as sparse edge lists because a dense
346-by-3,862 table is mostly zero. See `results/README.md` for the file map,
column definitions and interpretation limits.

The individual source-to-destination edges did not converge at the tested
particle densities. Use the matrices to reproduce the reported aggregate,
rank and broad spatial analyses; do not treat a single non-zero edge as a
confirmed ecological connection.

## Data provenance

Reef geometry was derived from the Great Barrier Reef features dataset
distributed through the Australian Ocean Data Network/Seamap Australia. The
hydrodynamic forcing came from the central-GBR ROMS model described by
Janekovic and Stanley (2027), DOI
[`10.1016/j.ocemod.2026.102825`](https://doi.org/10.1016/j.ocemod.2026.102825).

The EVD distributions were parameterised from Randall et al.; competency and
mortality follow Figueiredo et al. (2013), DOI
[`10.1890/12-0767.1`](https://doi.org/10.1890/12-0767.1). The repository is a
research record accompanying the thesis and should be read with the report's
full methods, citations and limitations.
