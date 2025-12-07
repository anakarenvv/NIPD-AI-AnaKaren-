# NIPD-AI-AnaKaren-

# 3D VGG16 fMRI Pain Classification – Replication & Input Pipelines

This repository contains the code used to partially replicate a 3D VGG16-based classifier for rodent fMRI pain states and to compare two input pipelines:

- **“KerasG” notebooks** → use the original **Keras Python generator**.
- **“tfdata” notebooks** → use the prototype **TensorFlow `tf.data` pipeline**.

The main experiments focus on **female CPH rats**, especially:
- Baseline vs Week 7 (BL vs W7)
- Baseline vs Week 1 (BL vs W1)

---

## Main files

- `my_data_generator.py`  
  Original Keras generator (Alan’s style). Used by notebooks with `KerasG` in the name.

- `tfdata_generator.py`  
  Prototype loader using `tf.data` and `tf.numpy_function`. Used by notebooks with `tfdata` in the name.

- `gradcam_utils.py`  
  Utility functions for Grad-CAM heatmaps, bounding boxes and visualizations.

- Notebooks (examples, adjust to your actual filenames):
  - `FEMALE_BLvsW7_KerasG.ipynb` → BL vs W7 using **Keras generator**.  
  - `FEMALE_BLvsW1_KerasG.ipynb` → BL vs W1 using **Keras generator**.  
  - `FEMALE_BLvsW7_tfdata.ipynb` → BL vs W7 using **tf.data** pipeline.

Whenever a notebook name contains:
- `KerasG` → it imports and uses `my_data_generator.py`.
- `tfdata` → it imports and uses `tfdata_generator.py`.

---

## Data folders you must have

The code assumes a structure like:

```text
RAW_ROOT/
  sub-XXX/ses-YY/func/..._task-rest_bold.nii
  sub-XXX/ses-YY/func/..._task-dist_bold.nii

RABIES_ROOT/
  preprocess_batch-001_rest/bold_datasink/commonspace_mask/
  preprocess_batch-001_rest/bold_datasink/commonspace_bold/
  preprocess_batch-002_rest/...
  preprocess_batch-00X/commonspace_bold/...
