# Analysis Scripts

This folder contains the research and benchmark scripts used to evaluate the iris recognition pipeline. Run commands from this folder unless a command says otherwise.

```bash
cd analysis
../.venv/bin/python <script>.py --help
```

Most scripts save generated files under `analysis/output/`.

Do not commit GT folders, datasets, generated output folders, or benchmark result dumps unless there is a specific reason and the data license allows it.

Install dependencies from the repository root with `pip install -r requirements.txt`; there is no separate analysis requirements file.

## Common Parameters

Most dataset scripts use some of these parameters:

- `--dataset` or `--datasets`: dataset name.
- `--dataset-path`: explicit dataset folder, when the default path should not be used.
- `--filters`: Python file containing a `filters` list. Defaults to project `../filters.py` when supported.
- `--rotation`: number of horizontal offsets to evaluate around zero.
- `--max-id`: maximum number of identities to sample.
- `--max-img-per-id`: maximum number of images per identity.
- `--seed`: deterministic random seed for subset sampling.
- `--output-name`: run name or output filename.

Use `--max-id`, `--max-img-per-id`, and `--seed` for bounded, repeatable experiments.

## Main Benchmark Scripts

### `benchmark_pipeline.py`

Runs the full pairwise benchmark over one or more datasets and prints/saves EER and related statistics.

Example:

```bash
../.venv/bin/python benchmark_pipeline.py \
  --datasets casia-v4-interval iitd mmu2 \
  --output-name baseline_rot21 \
  --rotation 21 \
  --rotation-step 4 \
  --max-id 80 \
  --max-img-per-id 15 \
  --seed 70 \
  --filters ../filters.py
```

Useful options:

- `--gt-mask`: use ground-truth masks from generated GT manifests instead of the segmenter. Supported local GT-mask datasets are `casia-v3-interval`, `casia-v4-interval`, and `iitd`. The command fails if a selected dataset has no GT manifest. GT masks are not stored in this repository; the masks used for these experiments came from [IRISEG-EP, WaveLab / Hofbauer14b](https://www.wavelab.at/sources/Hofbauer14b/).
- `--threshold`: evaluate a fixed Hamming-distance threshold in addition to EER.
- `--rotation-step N`: test every `N`th offset inside the centered `--rotation` range. For example, `--rotation 21 --rotation-step 4` tests offsets `-8, -4, 0, 4, 8`.
- `--parts N`: split each iriscode into `N` parts and use part-split average HD. This defaults to `--eliminate 0` and `--score hd`.
- `--score hd`: use selected-part average Hamming distance.
- `--score match-rotation`: classify by how many parts have matching best rotations.
- `--match-parts N`: with `--score match-rotation`, predict mated when at least `N` parts match.
- `--tolerance-offset N`: treat rotations within `+-N` pixels of the anchor rotation as matching, and keep only parts within that tolerance for HD scoring.

### `pairwise_iris_analysis.py`

Computes all pairwise scores for one dataset and plots mated/non-mated Hamming-distance distributions.

Example:

```bash
../.venv/bin/python pairwise_iris_analysis.py \
  --dataset-format casia-v4-interval \
  --output-name v4int_baseline_s70_rot21 \
  --rotation 21 \
  --max-id 80 \
  --max-img-per-id 15 \
  --seed 70 \
  --filters ../filters.py
```

Useful options:

- `--dataset-path`: override the dataset folder while keeping `--dataset-format` as the dataset layout selector.
- `--output-name`: name the generated distribution/ROC/EER figure.

### `compare_pipelines.py`

Compares two or more benchmark result JSON files and can plot a metric across datasets.

Example:

```bash
../.venv/bin/python compare_pipelines.py \
  baseline_rot21 part4_rot21 \
  --plot-metric eer \
  --datasets casia-v4-interval iitd mmu2
```

Useful plot metrics:

- `eer`
- `accuracy`
- `mated-classified-non-mated`
- `non-mated-classified-mated`
- `mated-classified-non-mated-rate`
- `non-mated-classified-mated-rate`
- `far`
- `frr`

## Part and Rotation Experiments

### `score_part_based_iriscode.py`

Used to test different numbers of parts and elimination settings, and to compare HD scoring against rotation-match scoring with an adjustable tolerance offset.

Example:

```bash
../.venv/bin/python score_part_based_iriscode.py \
  --dataset casia-v4-interval \
  --filters ../filters.py \
  --rotation 21 \
  --parts-range 4-10 \
  --eliminate-range 0-3 \
  --score hd \
  --max-id 80 \
  --max-img-per-id 20 \
  --seed 70 \
  --output-name v4int_part_sweep
```

Example:

```bash
../.venv/bin/python score_part_based_iriscode.py \
  --dataset casia-v4-interval \
  --filters ../filters.py \
  --rotation 21 \
  --parts-range 4-10 \
  --score match-rotation \
  --match-parts 1-6 \
  --tolerance-offset 3 \
  --max-id 80 \
  --max-img-per-id 20 \
  --seed 70 \
  --output-name v4int_match_rotation_sweep
```

### `rotation_EER.py`

Sweeps EER over rotation search ranges.

Example:

```bash
../.venv/bin/python rotation_EER.py \
  --axis horizontal \
  --dataset casia-v4-interval \
  --rotation 201 \
  --filters ../filters.py \
  --max-id 70 \
  --max-img-per-id 25 \
  --seed 70 \
  --output-name v4int_rotation_sweep
```

Use `--compare-methods` with `--score hd` to compare normal whole-iriscode HD against part-split average HD.

### `rotation_consistency_analysis.py`

Visualizes one mated and one non-mated comparison. It plots per-part HD across rotation offsets, so it is mainly for understanding how parts agree or disagree over rotation.

Example:

```bash
../.venv/bin/python rotation_consistency_analysis.py \
  --dataset casia-v4-interval \
  --filters ../filters.py \
  --rotation 21 \
  --seed 70 \
  --output-name v4int_rotation_consistency
```

### `radial_band_EER.py`

Measures EER separately for iris bands to study which radial regions contribute most.

Example:

```bash
../.venv/bin/python radial_band_EER.py \
  --dataset casia-v4-interval \
  --filters ../filters.py \
  --rotation 21 \
  --max-id 80 \
  --max-img-per-id 15 \
  --seed 70 \
  --output-name v4int_radial_bands
```

## Inspection and Debugging Scripts

### `interactive_hd_distribution.py`

Creates an interactive HD distribution. Clicking the plot shows image pairs near the selected Hamming distance.

Example:

```bash
../.venv/bin/python interactive_hd_distribution.py \
  --dataset casia-v3-twins \
  --max-id 80 \
  --max-img-per-id 3 \
  --selection-width 0.005 \
  --show-pairs 5 \
  --output-name v3twins_interactive_rot71 \
  --rotation 71 \
  --seed 70
```

### `mask_occlusion_tests.py`

Visualizes segmentation, occlusion masks, and source-mask overlays. Use this when checking whether invalid regions such as eyelids, eyelashes, reflections, or segmentation mistakes are being excluded correctly.

Example with single image:

```bash
../.venv/bin/python mask_occlusion_tests.py single \
  path/to/eye.png \
  --rotation 21 \
  --output-name single_mask_overlay
```

Example with multiple images:

```bash
../.venv/bin/python mask_occlusion_tests.py multiple \
  --dataset-format casia-v4-interval \
  --max-id 20 \
  --max-img-per-id 1 \
  --seed 70 \
  --output-name v4int_mask_overlay
```

### `score_vs_valid_bits.py`

Plots Hamming distance against the number of valid comparison bits. This is useful for checking whether low-valid-bit comparisons are unstable.

Example:

```bash
../.venv/bin/python score_vs_valid_bits.py \
  --dataset casia-v4-interval \
  --rotation 21 \
  --max-id 80 \
  --max-img-per-id 5 \
  --seed 70 \
  --output-name v4int_valid_bits
```

### `plot_pairwise_hd_scatter.py`

Plots pairwise comparisons as scatter points, with mated and non-mated comparisons separated by color.

Example:

```bash
../.venv/bin/python plot_pairwise_hd_scatter.py \
  --dataset casia-v4-interval \
  --filters ../filters.py \
  --rotation 21 \
  --max-id 80 \
  --max-img-per-id 5 \
  --seed 70 \
  --output-name v4int_pairwise_scatter
```
