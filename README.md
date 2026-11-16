# Deep Learning Approaches for Motion Correction in Spinal Cord Diffusion MRI and Functional MRI

This repository contains a slice-wise deep learning registration pipeline that estimates rigid in-plane translations (Tx, Ty) to motion-correct spinal cord **diffusion MRI (dMRI)** and **functional MRI (fMRI)** data.

Each 2D slice of each volume is registered to a reference slice (the temporal mean of the series) by a small convolutional network. Two networks are trained and chained at inference time:

- **Model 1 (coarse)**: trained on real data with additional synthetic random shifts (up to ±3 px), using a global + local NCC loss. It learns to recover large displacements.
- **Model 2 (refinement)**: trained on real data without synthetic shifts, using a local NCC loss only. It is applied iteratively after model 1 to refine the alignment.

The transforms are predicted on images cropped around the spinal cord, then transferred back and applied once to the original, uncropped image, so the corrected output keeps the input's grid, affine and header.

The fMRI method is also available in the [Spinal Cord Toolbox](https://spinalcordtoolbox.com/) (SCT v7.4). See [Citation](#citation) for the paper.

## Repository structure

The dMRI and fMRI pipelines live in separate, self-contained folders with the same layout:

```
moco-dl/
├── requirement.txt
├── dmri/
│   ├── crop_split_dataset.py        # subject-wise train/validation split + crop around the cord
│   ├── train_translation_first.py   # trains model 1 (coarse)
│   ├── train_translation_second.py  # trains model 2 (refinement)
│   ├── infer_adaptive.py            # two-stage motion correction of a 4D DWI series
│   ├── rigid_model_translation.py   # network (RigidRegistrationNet) + warping
│   └── coord_transform.py           # transfers a transform from the crop to the raw image
└── fmri/
    ├── download_data.sh             # downloads the OpenNeuro fMRI training data
    ├── files.txt                    # list of BOLD runs used for training
    ├── crop_split_dataset.py
    ├── cropped_data/split.csv       # train/validation split used for the published models
    ├── train_translation_first.py
    ├── train_translation_second.py
    ├── infer.py                     # two-stage motion correction of a 4D BOLD series
    ├── rigid_model_translation.py
    └── coord_transform.py
```

The scripts import their neighbours (`rigid_model_translation`, `coord_transform`) and use relative paths, so **run every command from inside `dmri/` or `fmri/`**.

## Dependencies

- Python 3.9+
- PyTorch, [MONAI](https://github.com/Project-MONAI/MONAI), NiBabel, NumPy, SciPy
- pandas and scikit-learn (dataset split)
- matplotlib, tqdm, [Weights & Biases](https://wandb.ai) (training logs and visualizations)
- `git` and [`git-annex`](https://git-annex.branchable.com/) (data download)
- `sc_crop`: command-line tool that crops an image around the spinal cord, called as `sc_crop -i INPUT -o OUTPUT`. It must be on your `PATH` for dataset preparation and inference. Another tool with the same interface can be used in dataset preparation via `--crop-cmd`.

Install the Python packages in your environment:

```bash
pip install -r requirement.txt
pip install pandas scikit-learn matplotlib
```

Training logs to Weights & Biases, so run `wandb login` once before training (or set `WANDB_MODE=offline`).

## Datasets

The models were trained on publicly available data from healthy participants, in BIDS format.

**fMRI**: 286 resting-state BOLD runs from 141 subjects across five OpenNeuro datasets: [ds004386](https://openneuro.org/datasets/ds004386/versions/1.1.2) (v1.1.2), [ds004616](https://openneuro.org/datasets/ds004616/versions/1.1.1) (v1.1.1), [ds005075](https://openneuro.org/datasets/ds005075/versions/1.0.1) (v1.0.1), [ds006729](https://openneuro.org/datasets/ds006729/versions/1.0.0) (v1.0.0) and [ds007932](https://openneuro.org/datasets/ds007932) (v1.1.0). The exact runs are listed in `fmri/files.txt`.

**dMRI**: DWI series from the [Spine Generic Public Database (multi-subject)](https://github.com/spine-generic/data-multi-subject).

## fMRI workflow

All commands below are run from `fmri/`.

### 1. Download the data

```bash
cd fmri
./download_data.sh /path/to/raw_data
```

This clones each OpenNeuro dataset at the version used for training into `/path/to/raw_data/<dataset>/` and downloads only the files listed in `files.txt` with `git annex get`. Files that fail to download are reported as `FAILED: ...` and the script continues. A different list can be passed as a second argument: `./download_data.sh /path/to/raw_data my_files.txt`.

### 2. Split and crop

```bash
python crop_split_dataset.py --data-root /path/to/raw_data --output cropped_data
```

- Splits the runs into training and validation **by subject** (all runs of a subject go to the same split), separately within each dataset so every dataset is represented in validation.
- Crops every run around the spinal cord with `sc_crop`.
- Writes `cropped_data/train/` and `cropped_data/validation/`, with files named `<dataset>_<original BIDS name>_crop.nii.gz`, and saves the split to `cropped_data/split.csv`.

| Option | Default | Description |
|---|---|---|
| `--data-root` | (required) | Folder given to `download_data.sh` |
| `--files` | `files.txt` | List of images, one `<dataset>/<path>` per line |
| `--output` | `cropped_data` | Where `train/` and `validation/` are written |
| `--val-size` | `0.2` | Fraction of subjects in validation |
| `--seed` | `42` | Random seed of the split |
| `--crop-cmd` | `sc_crop` | Crop command, called as `<cmd> -i INPUT -o OUTPUT` |

Images that already exist in the output folder are skipped, so the script can be re-run after a failure. The `split.csv` used for the published models is included in the repository.

### 3. Train

Training is configured by the constants at the top of each script (no command-line arguments). Before running, set at least:

- `DATA_DIR`: the folder containing `train/` and `validation/` (e.g. `cropped_data`)
- `CHECKPOINT_DIR`: where checkpoints are written

Other settings include `TARGET_SHAPE` (64×64 in-plane size after padding/cropping), `BATCH_SIZE`, `NUM_EPOCHS`, `LEARNING_RATE`, `NCC_WINDOW` and `WANDB_PROJECT`.

```bash
python train_translation_first.py    # model 1
python train_translation_second.py   # model 2
```

The two models are trained independently (model 2 does not start from model 1's weights) and can be trained in parallel. Each run saves `best_model.pt` (lowest validation loss) and `final_model.pt` in `CHECKPOINT_DIR`, and logs losses and example registrations (moving / fixed / warped / difference) to Weights & Biases.

During fMRI training, each slice at time *t* is paired with the same slice of the 10th volume (`REF_T = 9`) as the fixed image.

### 4. Inference

```bash
python infer.py \
    --input /path/to/sub-XX_task-rest_bold.nii.gz \
    --checkpoint1 /path/to/model1/best_model.pt \
    --checkpoint2 /path/to/model2/best_model.pt \
    --output /path/to/sub-XX_task-rest_bold_moco.nii.gz
```

The input is a raw (uncropped) 4D BOLD series. The script:

1. Crops it around the spinal cord with `sc_crop`.
2. **Stage 1**: predicts a translation for every (volume, slice) pair with model 1, against the temporal mean of the cropped series. A prediction is rejected (replaced by the identity) if it lowers the NCC between the slice and the reference.
3. **Stage 2**: applies model 2 iteratively (up to `--n-iter2` times) on the stage-1 result, with the same non-regression check, and stops early when the residual correction falls below `--tol2` or starts increasing.
4. Composes the stage-1 and stage-2 transforms, transfers them to the raw image coordinates, and warps each raw slice once.
5. Saves the corrected 4D volume on the raw grid (same affine and header), and prints the time spent in each step.

| Option | Default | Description |
|---|---|---|
| `--reference` | `average` | Stage-1 reference: temporal mean (`average`) or first volume (`first`) |
| `--reference2` | `recompute` | Stage-2 reference: `recompute` (temporal mean of the corrected series, recomputed at each iteration), `same` (stage-1 reference), or `running` (running mean of the previous corrected volumes) |
| `--n-iter2` | `3` | Maximum number of stage-2 iterations |
| `--tol2` | `1e-3` | Stage-2 convergence threshold |
| `--non-reg-tol` | `0.0` | Tolerated NCC decrease before a prediction is rejected |
| `--interpolation` | `nearest` | Interpolation of the final warp: `nearest`, `bilinear` or `bicubic` |
| `--cropped-path` | temporary file | Where to write the `sc_crop` output |
| `--keep-cropped` | off | Keep the cropped image instead of deleting it |

Inference runs on CPU by default (`DEVICE` at the top of the script).

## dMRI workflow

All commands below are run from `dmri/`. The pipeline is the same as for fMRI, with the differences listed in each step.

### 1. Download the data

Clone the Spine Generic dataset into a data folder and download its DWI series with `git-annex`:

```bash
mkdir -p /path/to/raw_data && cd /path/to/raw_data
git clone https://github.com/spine-generic/data-multi-subject.git
cd data-multi-subject
git annex get sub-*/dwi/
```

### 2. Split and crop

```bash
cd dmri
python crop_split_dataset.py --data-root /path/to/raw_data --output cropped_data_dwi
```

Same split and crop as for fMRI, with these differences:

- Without `--files`, every `*dwi.nii.gz` under `/path/to/raw_data/<dataset>/` is used. `sourcedata/` and `prepared/` folders are always skipped, and `derivatives/` folders are skipped unless `--include-derivatives` is given. With `--files`, only the `*dwi.nii.gz` entries of the list are kept.
- The `.bval`, `.bvec` and `.json` sidecars are copied next to each cropped image with the same name (cropping is spatial only, so the gradient table is unchanged). A warning is printed if a `.bval` or `.bvec` is missing next to the image, since it may be inherited from a higher level in BIDS.
- The default output folder is `cropped_data_dwi`.

### 3. Train

```bash
python train_translation_first.py    # model 1
python train_translation_second.py   # model 2
```

Configuration works as for fMRI (`DATA_DIR` defaults to `cropped_data_dwi`, `CHECKPOINT_DIR` to `dmri_chkpt/checkpoint_model1` and `dmri_chkpt/checkpoint_model2`). The dMRI training scripts add:

- **Mean reference**: each slice is registered to the mean of that slice across all volumes of the series.
- **Skipping of images with nothing to correct**: 3D images and series with fewer than `MIN_TIMEPOINTS` (2) volumes, such as a single b0 or reverse phase-encoding scan, are skipped.
- **Shifted validation set** (`val_shifted`): the validation data with the same kind of synthetic shift as in training, but with a fixed shift per pair so the loss is comparable across epochs. This loss is used for model selection and early stopping. A no-correction baseline is computed once for both validation sets and reported next to the losses.
- **Early stopping** after `EARLY_STOP_PATIENCE` (10) epochs without an improvement larger than `EARLY_STOP_MIN_DELTA` (1e-3) of the `val_shifted` loss (0 disables it).
- **Checkpoints**: `best_model.pt` (best `val_shifted` loss), `final_model.pt`, `latest.pt` (weights + optimizer state, overwritten every epoch) and a weights-only snapshot `epoch_XXX.pt` every `SAVE_EVERY` (10) epochs.
- **Resuming**: set `RESUME_FROM` to a `latest.pt` to continue an interrupted run.

### 4. Inference

```bash
python infer_adaptive.py \
    --input /path/to/sub-XX_dwi.nii.gz \
    --checkpoint1 /path/to/model1/best_model.pt \
    --checkpoint2 /path/to/model2/best_model.pt \
    --output /path/to/sub-XX_dwi_moco.nii.gz
```

Same two-stage procedure and options as the fMRI `infer.py`, with these differences:

- `--checkpoint2` is optional: without it, only stage 1 is applied.
- `--stage2-reference {average,first}` (default `average`) chooses the reference recomputed at each stage-2 iteration with `--reference2 recompute`: the temporal mean of the corrected series, or its first volume. In dMRI the first volume is usually a b0, whose contrast differs from the diffusion-weighted volumes.
- `--already-cropped` uses an image that was already cropped by `sc_crop` (e.g. from `cropped_data_dwi/validation/`) as is: the crop step is skipped and the output is on the cropped grid.
- Checkpoints can be weights-only files (`best_model.pt`, `epoch_XXX.pt`) or a `latest.pt`.
- 3D inputs and series with fewer than 2 volumes are rejected, since there is nothing to correct.
- The input's `.bval`, `.bvec` and `.json` are copied next to the output with the same name. The `.bvec` is copied unchanged: the correction is in-plane translation only, so gradient directions do not need to be rotated.

Example on a validation image:

```bash
python infer_adaptive.py --already-cropped \
    --input cropped_data_dwi/validation/<dataset>_sub-XX_dwi_crop.nii.gz \
    --checkpoint1 dmri_chkpt/checkpoint_model1/best_model.pt \
    --checkpoint2 dmri_chkpt/checkpoint_model2/best_model.pt \
    --output sub-XX_dwi_moco.nii.gz
```

## Model

`RigidRegistrationNet` (`rigid_model_translation.py`) takes the moving and fixed slices as a 2-channel 64×64 input and passes them through four strided 3×3 convolutions (16 → 128 channels, LeakyReLU), global average pooling and a linear layer that outputs the two translation parameters (Tx, Ty) in normalized coordinates. The last layer is initialized to zero, so the network starts from the identity. The translation is applied with `affine_grid` / `grid_sample`.

Before entering the network, each slice is clipped to its 1st–99th intensity percentiles, min-max normalized, and center-padded or cropped to 64×64.

The similarity loss combines:

- a **global NCC** computed only over pixels that are non-zero in both images (so zero-padding introduced by large shifts does not bias it), which gives a wide capture range for large displacements;
- a **local NCC** (MONAI `LocalNormalizedCrossCorrelationLoss`, 5×5 window), which is robust to intensity bias and drives fine sub-pixel alignment.

Model 1 uses both terms with equal weights; model 2 uses the local NCC only.

## Citation

If you use this code or the trained models in your work, please cite:

> Dagonneau T, Bédard S, Tucksinapinunchai N, Yarach U, Le Folgoc L, Gori P, Cohen-Adad J. Fast slice-wise deep learning motion correction for spinal cord functional MRI. arXiv:2611.14677 [cs.CV], 2026. https://arxiv.org/abs/2611.14677

```bibtex
@ARTICLE{Dagonneau2026-dv,
  title         = "Fast slice-wise deep learning motion correction for spinal
                   cord functional {MRI}",
  author        = "Dagonneau, Thomas and Bédard, Sandrine and Tucksinapinunchai,
                   Nontharat and Yarach, Uten and Le Folgoc, Loïc and Gori,
                   Pietro and Cohen-Adad, Julien",
  journal       = "arXiv [cs.CV]",
  month         =  nov,
  year          =  2026,
  eprint        = "2611.14677",
  archivePrefix = "arXiv",
  primaryClass  = "cs.CV",
  url           = "https://arxiv.org/abs/2611.14677"
}
```
