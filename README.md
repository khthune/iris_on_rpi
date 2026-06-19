# Iris Recognition on Raspberry Pi

This repository contains an iris recognition pipeline built around classical iriscode matching. The main pipeline segments an eye image, unwraps the iris annulus into a normalized band, applies Gabor-style filters, creates an iriscode and validity mask, and compares iriscodes with rotation-aware Hamming distance.

The project is mainly used for research on recognition performance, segmentation quality, filter design, rotation compensation, threshold selection, and Raspberry Pi suitability. The core pipeline is kept relatively small and interpretable, while most experiments live in [`analysis/`](analysis/).

## Repository Layout

- [`iris.py`](iris.py): core iris pipeline, segmentation loading, iris-band extraction, iriscode generation, and matching helpers.
- [`filters.py`](filters.py): default Gabor filter bank used by the pipeline.
- [`cli.py`](cli.py): command-line interface for generating, comparing, finding, and enrolling iriscodes.
- [`models/`](models/): local model files used by the pipeline.
- [`analysis/`](analysis/): benchmark scripts, plots, dataset loaders, filter experiments, part-based scoring, and research utilities.

See [`analysis/README.md`](analysis/README.md) for the analysis scripts, common commands, and experiment parameters.

## Setup

Create a virtual environment and install the Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

All project and analysis dependencies are kept in the root [`requirements.txt`](requirements.txt).

The active segmentation model is loaded by the pipeline code. If you need to override it for a local experiment, set `SEG_PATH` explicitly when running a command.

## CLI Usage

Generate an iriscode from one image:

```bash
python cli.py iris-gen path/to/eye.png -o iris_code.npy
```

Generate iriscodes from several images:

```bash
python cli.py iris-gens path/to/images/*.png
```

Compare two eye images:

```bash
python cli.py compare path/to/eye1.png path/to/eye2.png --rotation 21
```

Compare an image against a saved iriscode database:

```bash
python cli.py compare-iris-code path/to/eye.png iriscodes.npy --rotation 21
```

Find the best match in a saved iriscode database:

```bash
python cli.py find path/to/eye.png iriscodes.npy --rotation 21 --threshold 0.3
```

Enroll a new iriscode into an existing database:

```bash
python cli.py enroll path/to/eye.png iriscodes.npy
```

## Analysis Workflow

Most research work is done from the `analysis/` folder:

```bash
cd analysis
../.venv/bin/python benchmark_pipeline.py --datasets casia-v4-interval iitd --max-id 80 --max-img-per-id 15 --seed 70 --rotation 21
```

Generated plots, benchmark JSON files, score caches, manifests, and other run outputs should stay under `analysis/output/`.

## Datasets and Private Data

Dataset folders, generated outputs, and local model artifacts can be large or license-restricted. Do not commit datasets or generated benchmark outputs by default.

Ground-truth iris masks are not stored in this repository. The masks used for GT-mask experiments came from the IRISEG-EP dataset page:

- [IRISEG-EP, WaveLab / Hofbauer14b](https://www.wavelab.at/sources/Hofbauer14b/)

CASIA v3 Interval ground-truth masks can also be found here:

- [HalmstadUniversityBiometrics/Iris-Segmentation-Groundtruth-Database](https://github.com/HalmstadUniversityBiometrics/Iris-Segmentation-Groundtruth-Database)

Local GT-mask benchmarking is supported for `casia-v3-interval`, `casia-v4-interval`, and `iitd` when the private GT folders and generated manifests are present.

Generated analysis outputs should stay under:

- `analysis/output/`

## Supported Dataset Names

Many analysis scripts support these dataset identifiers:

- `casia-v1`
- `casia-v3-interval`
- `casia-v4-interval`
- `casia-distance`
- `casia-1000`
- `casia-v3-lamp`
- `casia-v3-twins`
- `iitd`
- `mmu`
- `mmu2`

## Notes

- The default filters file is [`filters.py`](filters.py).
- Rotation values are horizontal pixel offsets in the normalized iriscode search.
- Lower Hamming distance means two iriscodes are more similar.
