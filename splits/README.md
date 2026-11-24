# Saved train / test / validation split

The fixed 80/10/10 split of the 28,790 clean Galaxy Zoo images (`clean_gzdata5`) used for the HIWL rerun. These lists were saved from the reproduction notebook (`read_split_data`, seed 0).

| File | Images | Used for |
|---|---|---|
| `train_set.csv` | 23,031 (80%) | Training |
| `test_set.csv` | 2,878 (10%) | Checking and picking the best model during training (the authors' *val*) |
| `validation_unseen_set.csv` | 2,881 (10%) | Unseen hold-out, used only for the final results (the authors' *test*) |

Each row is `path,label` (label 0 = bet/IBS, 1 = cig/CSS, 2 = edg/EO, 3 = rou/CRS, 4 = spi/SPI). No image appears in more than one file.

**Why it is saved:** the authors' split depends on the order the computer lists files, which changes between machines and Kaggle sessions. To use these files instead, set the environment variable before running any script in `scheme/`:

```bash
export HIWL_SPLIT_DIR=/path/to/HIWL-DS340W/splits
```

`read_split_data` in `scheme/utils.py` then loads these lists (mapping each `class/file.jpg` onto the given `--data-path5`) instead of re-shuffling. Without the variable, the code behaves exactly like the original.
