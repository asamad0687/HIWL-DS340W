# Notes on Running This Code (Reproduction Environment)

This repo is forked from the original authors' code for:
"Galaxy Image Classification Using Hierarchical Data Learning with Weighted Sampling and Label Smoothing" (Ma et al., MNRAS 2023).

I ran this code in a **Kaggle Notebook** (GPU: T4 x2) to reproduce the parent paper's data preparation and training pipeline for DS 340W (Group [XX]).

## Environment Setup Notes

1. **Dataset:** Downloaded from the official Kaggle competition: [Galaxy Zoo - The Galaxy Challenge](https://www.kaggle.com/competitions/galaxy-zoo-the-galaxy-challenge/data)

2. **`select cleandata.py`:** The original file has hardcoded local Windows paths (`F:\dataSet\...`). To run in Kaggle, these three variables were updated to point to the Kaggle input/working directories:
   - `gzimg_path`
   - `gzlabel_path`
   - `gzclean_path`

3. **`pytorchtools.py`:** This file was missing from the original repo. It is a standard, widely-used `EarlyStopping` utility (based on the common `early-stopping-pytorch` implementation) that `train_efficientnet.py` depends on but was not included in the original upload. I added it here so the code runs as intended.

4. **Running `train_efficientnet.py`:** Since pretrained weights and the original local dataset path aren't available in this environment, the script was run with the following command-line arguments (no changes to the script's internal logic):
