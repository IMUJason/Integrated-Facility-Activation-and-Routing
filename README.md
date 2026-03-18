# Integrated Facility Activation and Routing for C&D Waste

This repository accompanies the computational package behind the study on integrated facility activation and routing for C&D waste.

## Scope

This public release is limited to the asset set that matches the current `Waste Management` revision:

- `instances/`: anonymized benchmark instances used by the public computational package
- `code/`: solver and experiment scripts for the released benchmark instances
- `results/`: aggregated experiment tables and summary JSON files reported in the manuscript

The repository intentionally excludes:

- current manuscript source files and compiled submission PDFs
- raw regulatory filings
- geocoded project records with original identifiers
- preprocessing files that contain project- or location-level administrative information
- superseded TRSC drafts, proofs, and intermediate working files

## Anonymization

The released benchmark instances preserve demands, capacities, time windows, route-planning parameters, and geometry needed for the computational study, but original project names and disposal-site identifiers have been replaced with anonymized labels such as `project_001` and `facility_001`.

## Reproduction Boundary

The public package starts from the released benchmark instances in `instances/`.

- `code/run_pipeline.py` reproduces the main computational pipeline from these public instances.
- Raw-data preprocessing from administrative filings is not part of the public release because the retained source records are not publicly redistributable.

## Environment

The public scripts were prepared around the following Python dependencies:

- `numpy`
- `docplex`
- `torch`

Additional details are listed in `requirements.txt`.

## Repository Notes

See `docs/repository_scope.md` for the release boundary and asset-selection rationale.
