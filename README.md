
# 3D VGG16 fMRI Pain Classification – Replication & Input Pipelines

This repository contains the code used to partially replicate a 3D VGG16-based classifier for rodent fMRI pain states and to compare two input pipelines:

- **“KerasG” notebooks** → use the original **Keras Python generator**.
- **“tfdata” notebooks** → use the prototype **TensorFlow `tf.data` pipeline**.
----

## Main files

- `my_data_generator.py`  
  Original Keras generator. Used by notebooks with `KerasG` in the name.

- `tfdata_generator.py`  
  Prototype loader using `tf.data` and `tf.numpy_function`. Used by notebooks with `tfdata` in the name.

- `gradcam_utils.py`  
  Utility functions for Grad-CAM heatmaps, bounding boxes and volume visualizations.

- Notebooks

  - `FEMALE_BLvsW7_tfdata.ipynb` → BL vs W7 using **tf.data** pipeline.
  - `FEMALE_BLvsW7_KerasG.ipynb` → BL vs W7 using **Keras generator**.  
  - `FEMALE_BLvsW1_KerasG.ipynb` → BL vs W1 using **Keras generator**.  .
  - MALE_BLvsW7_KerasG.ipynb` → BL vs W7 using **Keras generator**.  
  -`MALE_BLvsW1_KerasG.ipynb` → BL vs W1 using **Keras generator**.  
  

Whenever a notebook name contains:
- `KerasG` → it imports and uses `my_data_generator.py`.
- `tfdata` → it imports and uses `tfdata_generator.py`.

---

## Data layout and `FILES_and_LABELS`

The helper class `FILES_and_LABELS` is responsible for building lists of NIfTI files and labels from:

- **Raw BIDS-like data** (`RAW_ROOT`)
- **RABIES preprocessed outputs** (`RABIES_ROOT`)


## Optional: Tracking experiments with Weights & Biases (WandB)

Some notebooks are configured to log training metrics and artifacts to [Weights & Biases](https://wandb.ai) (WandB) using:

Using WandB is optional, but recommended if you want to:

- Keep a history of loss, accuracy, AUC, etc.
- Compare runs (different folds, hyperparameters, epochs).
- Store Grad-CAM images and other plots as artifacts.

To use WandB you need to:

- Create a free account at https://wandb.ai
- Install WandB in your environment:

- ### Optional: Log experiments with Weights & Biases (wandb)

To enable experiment tracking with [Weights & Biases](https://wandb.ai) you need to:

1. Create a free account at [https://wandb.ai](https://wandb.ai).
2. Install `wandb` in your environment:

3. Log in once form a terminal or notebook
   wand login

4. Then paste your API key (available in your wandb account settings).

After this one-time setup, the notebooks will automatically create a new run each time wandb.init(...) is called and log metrics during training
   

   








