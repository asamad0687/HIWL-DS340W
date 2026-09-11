# Notes on Running This Code (Reproduction Environment)

This repo is forked from the original authors' code for:
"Galaxy Image Classification Using Hierarchical Data Learning with Weighted Sampling and Label Smoothing" (Ma et al., MNRAS 2023).

I ran this code in a **Kaggle Notebook** (GPU: T4 x2) to reproduce the parent paper's data preparation and training pipeline for DS 340W (Group [XX]).

## Environment Setup Notes

1. **Dataset:** Downloaded from the official Kaggle competition: [Galaxy Zoo - The Galaxy Challenge](https://www.kaggle.com/competitions/galaxy-zoo-the-galaxy-challenge/data)

2. **`select cleandata.py`:** This file is left unmodified in this repository, exactly as published by the original authors. However, it contains hardcoded local Windows file paths that do not exist in a Kaggle environment. To actually run it, the following three path variables were substituted at runtime (path values only — no logic was changed):

   **Original (as published):**
```python
   gzimg_path = r'F:\dataSet\galaxy-zoo-the-galaxy-challenge\images_training_rev1'
   gzlabel_path = r'F:\dataSet\galaxy-zoo-the-galaxy-challenge\training_solutions_rev1.csv'
   gzclean_path = r'F:\dataSet\clean gzdata5'
```

   **Substituted for Kaggle environment:**
```python
   gzimg_path = '/kaggle/input/notebooks/kingjuno/galaxy-zoo/images_training_rev1'
   gzlabel_path = '/kaggle/input/notebooks/kingjuno/galaxy-zoo/training_solutions_rev1.csv'
   gzclean_path = '/kaggle/working/clean_gzdata5'
```

   This substitution was applied to a local copy of the file at runtime inside the Kaggle notebook; the version of `select cleandata.py` committed to this repository remains the original, unmodified file from the authors.

3. **`pytorchtools.py`:** This file was missing from the original repo. It is a standard, widely-used `EarlyStopping` utility (based on the common `early-stopping-pytorch` implementation) that `train_efficientnet.py` depends on but was not included in the original upload. I added it here so the code runs as intended.

4. **Running `train_efficientnet.py`:** Since pretrained weights and the original local dataset path aren't available in this environment, the script was run with the following command-line arguments (no changes to the script's internal logic):
