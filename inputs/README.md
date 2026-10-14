# Processed spatial input

`model_inputs.gpkg` is the processed spatial input used by the model. Its main
layers are:

- `reefs`: all 3,862 destination reef polygons;
- `sources`: clipped source polygons for the prepared domains, including the
  346 Domain C sources used in the report;
- `domains`: release-domain geometries.

The source data are not modified by the scripts in this repository. Geometry
repair and clipping provenance are stored as attributes in the GeoPackage.
Raw ROMS, GFS and source shapefile data are not included.
