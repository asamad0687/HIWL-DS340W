# Notes on Running This Code (Reproduction Environment)

This repo is forked from the original authors' code for:
"Galaxy Image Classification Using Hierarchical Data Learning with Weighted Sampling and Label Smoothing" (Ma et al., MNRAS 2023).

I ran this code in a **Kaggle Notebook** (GPU: T4 x2) to reproduce the parent paper's data preparation and training pipeline for DS 340W (Abdul Samad, solo project).

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

```bash
python train_efficientnet.py --data-path5 /kaggle/working/clean_gzdata5 --weights4 "" --weights2 "" --os linux --device cuda:0 --num-model B1
```

   - `--weights4 ""` / `--weights2 ""`: the authors' EfficientNet-B1 pretrained weights are only hosted on Baidu Pan (see `pretrain url.txt`), so both models train from scratch.
   - `--os linux`: turns on DataLoader workers (on Windows the script sets them to 0).
   - `--num-model B1`: the EfficientNet-B1 backbone the paper uses for its main HIWL results (240x240 input).

5. **`evaluate_metrics.py` (added):** `train_efficientnet.py` only prints the overall test accuracy. To compare against the paper's Table 3 (confusion matrix) and Table 4 (recall / precision / F1), this script reads `predicts/B1/test_pred_all.pth`, rebuilds the same seeded 8:1:1 split with the authors' `read_split_data`, and prints the metrics next to the paper's values. It doesn't retrain anything or change the authors' code.

```bash
python evaluate_metrics.py --data-path5 /kaggle/working/clean_gzdata5 --num-model B1
```

6. **Split naming:** in the authors' code, the split they call `val` is used to pick the best model pair, and the split they call `test` gives the final reported accuracy. The notebook's last data cell saves them as `test_set.csv` (authors' val) and `validation_unseen_set.csv` (authors' test, the one the paper reports).

