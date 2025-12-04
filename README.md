# Deep Learning Approaches for Motion Correction in Spinal Cord Diffusion MRI and Functional MRI
This repository introduces a DenseNet-based slice-wise regressor that estimates rigid in-plane translations (Tx, Ty) for motion correction in diffusion MRI (dMRI) and functional MRI (fMRI).

## Overview
1. **Generate motion-augmented data** using `augmentation.py`  
   → Simulate slice-wise rigid motion artifacts (only in-plane motion) in dMRI and fMRI data.

2. **Preprocess the dataset** using `preprocessing.py`  
   → Prepare mean dMRI/fMRI volumes as the taeget volume, perform spinal cord segmentation and create mask along spinal cord via **Spinal Cord Toolbox (SCT)**.

3. **Prepare datasets for training** using `dataset_preparation.py`  
   → Organize the dataset structure and split into training, validation, (in .pt format) and testing sets.

4. **Train the DenseNet-based deep learning model** using `moco_main.py`  
   → Learn rigid slice-wise translations (Tx, Ty) for motion correction across time.

5. **Evaluate and test model performance** using `test_model.py`  
   → Apply the trained model checkpoint to new data, correct motion, and export the 4D motion-corrected volumes as well as Tx and Ty translation parameter.

## Dependencies
The primary dependencies for this project are:
*   [Spinal Cord Toolbox (SCT)](https://spinalcordtoolbox.com/): Required for the `preprocessing.py` script.
*   Python 3.9
*   PyTorch Lightning
*   [MONAI](https://github.com/Project-MONAI/MONAI)
*   NiBabel
*   scikit-image
*   PyYAML
*   Weights&Biases (wandb) — for training visualization and experiment tracking

You can use `requirement.txt` to install dependencies inside your conda environment:
```bash
pip install -r requirement.txt
```

## Dataset for training
This project used publicly available healthy-participant datasets: 
*   [Spine Generic Public Database (multi-subject)](https://github.com/spine-generic/data-multi-subject) for dMRI (n = 267)
*   OpenNeuro Database ([ds004386](https://openneuro.org/datasets/ds004386/versions/1.1.2), [ds004616](https://openneuro.org/datasets/ds004616/versions/1.1.1), [ds005075](https://openneuro.org/datasets/ds005075/versions/1.0.1), [ds006729](https://openneuro.org/datasets/ds006729/versions/1.0.0)) for fMRI (n = 232).

The dataset follows the BIDS convention. For example:

```
sub-01/
├── ses-01/
│   ├── anat/
│   │   ├── sub-01_ses-01_T2w.json
│   │   ├── sub-01_ses-01_T2w.nii.gz
│   ├── func/
│   │   ├── sub-01_ses-01_task-rest_bold.json
│   │   ├── sub-01_ses-01_task-rest_bold.nii.gz
│   ├── dwi/
│   │   ├── sub-01_dwi.bval
│   │   ├── sub-01_dwi.bvec
│   │   ├── sub-01_dwi.json
│   │   ├── sub-01_dwi.nii.gz
```

## Model Architecture and Training
This project used the `DenseNet` model implemented in PyTorch Lightning (`moco_main.py`).

*   **Backbone (`DenseNetRegressorSliceWise`)**: A DenseNet-based architecture that takes a pair of slices (motion-augmented and target) and predicts the required translation parameter (Tx, Ty) to align them. It processes the entire volume slice-by-slice.

*   **Warping (`RigidWarp`)**: A module that converted Tx and Ty translation parameters into sampling grids and applied to the motion-augmented volumes using a bilinear interpolation.

*   **Loss Function**: A composite loss function is used to guide the training:
    *   **Similarity Loss**: A weighted combination of Global Normalized Cross-Correlation (GNCC) and L2 (MSE) loss to maximize the similarity between the motion-corected and target volumes within the spinal cord region.
    *   **Smoothness Regularization**: Penalizes large translations and encourages smooth transitions of translation parameters between adjacent slices and time points.
 
## End-to-End Workflow
The following steps guide you through the process, from data preparation to model inference. The scripts are designed to work with a **BIDS**-like directory structure.

### 1. Data Augmentation
Simulate slice-wise rigid motion artifacts in your clean dMRI or fMRI data to create the "motion-augmented" volumes for training. The script automatically detects files `*dwi.nii.gz` for dMRI or `*bold.nii.gz` for fMRI.

**Usage:**
```bash
python augmentation.py /path/to/your_data <dmri|fmri>
```
*   `/path/to/your_data`: The root directory containing subject folders
*   `mode`: Specify `dmri` or `fmri`

This generates an `aug_*.nii.gz` file for each subject.

### 2. Preprocessing with SCT
Prepare the necessary reference (averaged or target) volumes and masks using the Spinal Cord Toolbox (SCT).

*   **For dMRI**: Separates b0/dwi volumes, creates a mean dwi/b0 image, segments the cord, creates a mask.
*   **For fMRI**: Computes a mean volume across time as a target image, segments the cord, and creates a mask.

**Usage:**
```bash
python preprocessing.py /path/to/your_data <dmri|fmri>
```
*   `/path/to/your_data`: The root directory containing subject folders
*   `mode`: Specify `dmri` or `fmri`

### 3. Dataset Preparation
Convert the preprocessed NIfTI files into PyTorch tensors (`.pt`), split subjects into training (80%), validation (10%), and testing (10%) sets, and generate a `dataset.json` file to index the dataset. This organizes the data into a `prepared` subfolder.

**Usage:**
```bash
python dataset_preparation.py /path/to/your_data <dmri|fmri>
```
*   `/path/to/your_data`: The root directory containing subject folders
*   `mode`: Specify `dmri` or `fmri`

This creates a `prepared/<mode>_dataset` directory containing the structured dataset and the `dataset.json` index file.

### 4. Model Training
Train the DenseNet-based model using the prepared dataset. The script uses PyTorch Lightning for training and supports logging with Weights & Biases.

**Usage:**
```bash
python moco_main.py /path/to/project_base /path/to/prepared_dataset <run_name1> <run_name2>(opt)
```
*   `/path/to/project_base`: Base directory containing the script (.py) and trained_weights folder
*   `/path/to/prepared_dataset`: Path to the output directory from Step 3 (e.g., `/path/to/data/prepared/dmri_dataset`)
*   `<run_name1>`: An identifier for this training run. The best model checkpoint will be saved as `<run_name1>.ckpt`
*   `<run_name2>`: If provided, in case of fine-tune from `<run_name1>`, it will be save under new model checkpoint `<run_name2>.ckpt`

### 5. Inference
Apply a trained model checkpoint to a test dataset to perform motion correction. This script loads the test data, runs the model, and saves the 4D motion-corrected volume along with the predicted translation parameters (Tx and Ty).

**Usage:**
```bash
python test_model.py /path/to/testing /path/to/trained_weight.ckpt
```
*   `/path/to/testing`: The path to the `testing` folder inside your prepared dataset directory
*   `/path/to/trained_weight.ckpt`: The full path to the trained model checkpoint file

**Outputs:**
For each test subject, the following files are saved in their respective `func` or `dwi` subdirectories:
*   `moco_*.nii.gz`: The 4D motion-corrected NIfTI volume
*   `*_Tx.nii.gz`: The predicted translation parameters in the x-direction
*   `*_Ty.nii.gz`: The predicted translation parameters in the y-direction

### 6. Quantitative evaluation (optional)
After motion correction, you can quantitatively evaluate the improvement in image quality using `metrics.py`.  
This script automatically computes voxel-wise and temporal metrics before and after correction for both **dMRI** and **fMRI** datasets.
- Computes the following metrics:
  - **RMSE (Root Mean Squared Error)** — evaluates voxel-wise similarity
  - **SSIM (Structural Similarity Index)** — measures 3D structural fidelity  
  - **tSNR (temporal Signal-to-Noise Ratio)** — evaluates signal stability across time *(fMRI only)*  
  - **DVARS (temporal derivative of RMS variance)** — quantifies temporal signal fluctuation *(fMRI only)*

**Usage:**
```bash
python metrics.py /path/to/testing <dmri|fmri>
```
*   `/path/to/testing`: The path to the `testing` folder inside your prepared dataset directory (same as in 5.Inference)
*   `mode`: Specify `dmri` or `fmri`

**Outputs:**
A summary CSV file (`dmri_metrics.csv` or `fmri_metrics.csv`): the reported values for each subject are averaged across timepoints.
