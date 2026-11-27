# HIWL Galaxy Classification: DS 340W Reproduction

**Course:** DS 340W, The Pennsylvania State University
**Student:** Abdul Samad (afs6958@psu.edu), solo project

This repo is a fork of the authors' code for the parent paper:

> Ma, X., Li, X., Luo, A., Zhang, J., & Li, H. (2023). *Galaxy image classification using hierarchical data learning with weighted sampling and label smoothing.* Monthly Notices of the Royal Astronomical Society, 519(3), 4765–4779. https://doi.org/10.1093/mnras/stac3770

Original code: https://github.com/xrli/HIWL

It reproduces HIWL (hierarchical classification + weighted sampling + label smoothing with an EfficientNet-B1 backbone) on Kaggle, and is the base for my novel approach: confidence scoring and prediction stability (see [`novel/`](novel/)).

---

## Dataset

**Galaxy Zoo – The Galaxy Challenge** (Kaggle competition)
https://www.kaggle.com/competitions/galaxy-zoo-the-galaxy-challenge/data

- 61,578 labelled galaxy images (`images_training_rev1`) plus volunteer vote fractions (`training_solutions_rev1.csv`)
- `select cleandata.py` picks the **28,790 "clean" images** in 5 classes, following the paper's Table 1 thresholds:

| Class | Folder | Paper name | Clean images |
|---|---|---|---|
| Completely round smooth | `rou` | CRS | 8,434 |
| In-between smooth | `bet` | IBS | 8,069 |
| Cigar-shaped smooth | `cig` | CSS | 578 |
| Edge-on | `edg` | EO | 3,903 |
| Spiral | `spi` | SPI | 7,806 |

- Split **8:1:1** into train / validation / test (23,031 / 2,878 / 2,881), seeded with `random.seed(0)` in `utils.read_split_data`

---

## How to run (Kaggle)

The full step-by-step notebook is [`notebooks/ds340w_hiwl_reproduction.ipynb`](notebooks/ds340w_hiwl_reproduction.ipynb).

1. Create a Kaggle notebook, turn on **GPU (T4 x2)**, and add the Galaxy Zoo data as an input.
2. Clone this repo and `cd HIWL-DS340W/scheme`.
3. **Clean sample selection:** run `select cleandata.py` with the three Windows paths swapped for Kaggle paths (the notebook does this at runtime; the file itself is unchanged).
4. **Train HIWL (EfficientNet-B1):**
   ```bash
   python train_efficientnet.py --data-path5 /kaggle/working/clean_gzdata5 \
       --weights4 "" --weights2 "" --os linux --device cuda:0 --num-model B1
   ```
5. **Evaluate against the paper:**
   ```bash
   python evaluate_metrics.py --data-path5 /kaggle/working/clean_gzdata5 --num-model B1
   ```
   This prints overall accuracy, per-class recall / precision / F1 next to the paper's Table 4, and the confusion matrix. It also saves `results/metrics_B1.csv`.

Full environment notes (path changes, the missing `pytorchtools.py`, argument choices): [`scheme/RUNNING_NOTES.md`](scheme/RUNNING_NOTES.md).

---

## Results: reproduction vs. parent paper

HIWL with EfficientNet-B1. Kaggle run on GPU T4 x2, about 8.5 hours (30,495 s) end to end.

| Metric | Paper (Ma et al. 2023) | My reproduction |
|---|---|---|
| **Test accuracy** (2,881 images) | **96.32%** | **96.49%** |
| Best validation accuracy, combined hierarchical model | not reported | 96.42% |

From the training log: `acc_combine_best: 0.9642, acc_combine_best_index: (0, 0), test_acc: 0.9649`

The reproduction comes within **+0.17 percentage points** of the paper's overall accuracy.

**Paper's per-class results (Table 4), for reference:**

| Class | Recall | Precision | F1 |
|---|---|---|---|
| CRS | 0.9633 | 0.9679 | 0.9656 |
| IBS | 0.9628 | 0.9499 | 0.9563 |
| CSS | 0.7069 | 0.7855 | 0.7455 |
| EO  | 0.9616 | 0.9641 | 0.9629 |
| SPI | 0.9834 | 0.9834 | 0.9834 |
| Avg | 0.9156 | 0.9302 | 0.9227 |

### Rerun on the saved split (final numbers)

The authors' split depends on the folder listing order, so it can't be rebuilt in a new session. To get results on a fixed split, HIWL was retrained once on the saved split in [`splits/`](splits/) ([`notebooks/ds340w_hiwl_rerun.ipynb`](notebooks/ds340w_hiwl_rerun.ipynb), about 8.9 hours on GPU T4 x2). Training log: `acc_combine_best: 0.9680, acc_combine_best_index: (0, 0), test_acc: 0.9524`.

| Class | Recall (paper) | Precision (paper) | F1 (paper) |
|---|---|---|---|
| CRS | 0.9704 (0.9633) | 0.9523 (0.9679) | 0.9613 (0.9656) |
| IBS | 0.9517 (0.9628) | 0.9540 (0.9499) | 0.9529 (0.9563) |
| CSS | 0.6034 (0.7069) | 0.5833 (0.7855) | 0.5932 (0.7455) |
| EO  | 0.9309 (0.9616) | 0.9381 (0.9641) | 0.9345 (0.9629) |
| SPI | 0.9706 (0.9834) | 0.9870 (0.9834) | 0.9787 (0.9834) |
| Avg | 0.8854 (0.9156) | 0.8830 (0.9302) | 0.8841 (0.9227) |

Overall test accuracy on the saved split is **95.24%** (paper: 96.32%). These are the numbers the novel approach builds on; see [`novel/`](novel/).

**Why small differences are expected:** the authors' pretrained EfficientNet-B1 weights are only on Baidu Pan, so both models here train **from scratch** (`--weights4 "" --weights2 ""`). On top of that, GPU nondeterminism and Kaggle's newer PyTorch / torchvision versions can shift results a little. As in the paper, the cigar-shaped class (CSS, only 58 test images) is the hardest and the most variable.

---

## Kaggle notebooks

- Reproduction: [DS340W](https://www.kaggle.com/code/abduls135/ds340w)
- Rerun on the saved split + novel approach (final results): [DS340W HIWL Rerun](https://www.kaggle.com/code/abduls135/ds340w-hiwl-rerun)
- First (preliminary) novel-approach run: [DS340W Novel Approach](https://www.kaggle.com/code/abduls135/ds340w-novel-approach)

---

## Repo layout

```
HIWL-DS340W/
├── select cleandata.py          # Step 1: clean sample selection (authors', unchanged)
├── scheme/                      # HIWL models (with the scheme), used for the reproduction
│   ├── train_efficientnet.py    # Step 2: hierarchical training (authors')
│   ├── model_efficientnet.py    #   backbone (authors')
│   ├── utils.py                 #   split / train / evaluate helpers (authors')
│   ├── my_dataset.py            #   dataset + label grouping (authors')
│   ├── pytorchtools.py          #   EarlyStopping (added; missing from original upload)
│   ├── evaluate_metrics.py      # Step 3: per-class metrics vs. paper (added)
│   └── RUNNING_NOTES.md         #   reproduction environment notes (added)
├── noscheme/                    # Baseline models without HIWL (authors')
├── notebooks/
│   ├── ds340w_hiwl_reproduction.ipynb   # Kaggle notebook, run top to bottom
│   └── ds340w_hiwl_rerun.ipynb          # Retrain on the saved split + novel approach (final results)
├── novel/                       # Novel approach: confidence scoring + prediction stability (final results in novel/results)
├── splits/                      # Saved train / test / validation split used for the final results
├── requirements.txt             # authors' original versions (Kaggle's preinstalled versions were used)
└── pretrain url.txt             # authors' pretrained-weight links
```

The pipeline is split into separate stages: **data selection → training → evaluation → novel layer**. That way the novel-approach modules can plug in after evaluation without rewriting the authors' training code.

---

## Original authors' README

This repo contains the code for our paper *Galaxy Image Classification using Hierarchical Data Learning with Weighted Sampling and Label Smoothing*.

### Requirements

- numpy==1.19.5
- matplotlib==3.2.1
- tqdm==4.56.0
- torch>=1.7.1
- torchvision>=0.8.2

### Select cleandata

- Selecting clean data from the original data downloaded from https://www.kaggle.com/competitions/galaxy-zoo-the-galaxy-challenge/data.
```
select cleandata.py
```

### Noscheme
- Training the models without HIWL
```
train_dieleman.py
train_vgg.py
train_googlenet.py
train_resnet26.py
train_resnet.py
train_efficientnet.py
train_vit.py

```


### Scheme
- Training the models with HIWL
```
train_dieleman.py
train_vgg.py
train_googlenet.py
train_resnet26.py
train_resnet.py
train_efficientnet.py
train_vit.py

```



### Usage

- Training a new model:
1. Selecting cleandata
2. training new model from scheme or noscheme
  ```shell
  python select cleandata.py
  python train_efficientnet.py
  ```

### Citation

- If you found this code useful please cite our paper: 
- 

