# Novel approach: confidence scoring and prediction stability

**Working title:** *Knowing When the Model Is Unsure: Confidence Scoring and Prediction Stability for Galaxy Morphology Classification*

This folder holds the DS 340W novel-approach code. It's built **on top of** the HIWL pipeline in `../scheme/` and only changes the **inference / evaluation stage**: instead of keeping just HIWL's final class, it keeps the model's probabilities and uses them to flag the galaxies the model is unsure about, so a human can review them.

| File | What it is |
|---|---|
| `hiwl_uncertainty.py` | The novel-approach script (loads the trained HIWL models, scores every test galaxy, writes the results) |
| `results/` | **Final results** from the rerun on the saved split (see below) |
| `ds340w_novel_approach.ipynb` | The first, preliminary Kaggle run (superseded by the rerun; kept for the record) |

The final results come from **`../notebooks/ds340w_hiwl_rerun.ipynb`** (Kaggle: [DS340W HIWL Rerun](https://www.kaggle.com/code/abduls135/ds340w-hiwl-rerun)), which retrains HIWL on the saved split and then runs this script.

## How it works

HIWL is hierarchical. **Model 4** sorts a galaxy into 4 groups (in-between smooth, round, spiral, or "cigar **or** edge-on"), and **Model 2** splits that last group into cigar vs. edge-on. The script combines both models into one 5-class probability per galaxy:

- P(IBS), P(CRS), P(SPI) come from Model 4
- P(CSS) = P4(cigar-or-edge-on) × P2(cigar), and P(EO) = P4(cigar-or-edge-on) × P2(edge-on)

The predicted class follows the authors' rule in `utils.evaluateall` exactly, so the uncertainty layer never changes what HIWL predicts.

**Five ways to flag uncertain predictions** (the comparison):

| Method | Idea |
|---|---|
| Max probability (MSP) | How high is the predicted class's probability? |
| Margin | How far ahead is the top class of the second one? |
| Entropy | How spread out are the probabilities across all 5 classes? |
| Stability | Re-run the galaxy with 8 exact flips/rotations. How often does the answer stay the same? |
| TTA-averaged confidence | MSP after averaging the 8 flipped/rotated predictions |

**Human review budget:** sort galaxies from least to most confident, send only the bottom 5%, 10% or 20% to a human, and count how many of the model's mistakes that catches. **AUROC** measures how well each score separates wrong predictions from correct ones (0.5 = random, 1.0 = perfect).

## How to run (Kaggle)

1. New notebook → import `../notebooks/ds340w_hiwl_rerun.ipynb`.
2. Add Input: the **DS340W** reproduction notebook's output (for `clean_gzdata5`). GPU T4 x2, Internet on.
3. Save & Run All. Training takes about 8.5–9 hours; the uncertainty step takes about 2 minutes.

The notebook sets `HIWL_SPLIT_DIR` so every script uses the saved split in `../splits/`. It also picks the saved checkpoints that reproduce HIWL's own final test predictions (100% agreement), and runs:

```bash
python ../novel/hiwl_uncertainty.py --data-path5 <clean_gzdata5> --weights-dir <best_pair> \
    --ref-pred predicts/B1/test_pred_all.pth --out-dir results_novel
```

To skip training, add the *DS340W HIWL Rerun* notebook's output as Input and run only the last step with `--weights-dir` pointing to its `best_pair/` folder.

## Results (2,881 unseen test galaxies, 137 errors)

All test galaxies come from `splits/validation_unseen_set.csv` and were never used for training or model selection.

**HIWL itself (retrained):** accuracy **95.24%** (paper: 96.32%). Per-class metrics are in `results/hiwl_metrics_B1.csv` and the confusion matrix in `results/hiwl_confusion_matrix.csv`. Cigar-shaped recall is the lowest (0.60 vs. 0.71 in the paper), mostly confused with edge-on.

**Uncertainty methods:**

| Method | AUROC (error detection) | Errors caught, review 5% | Review 10% | Review 20% |
|---|---|---|---|---|
| Random review (baseline) | 0.500 | 5.0% | 10.0% | 20.0% |
| Max probability (MSP) | 0.931 | 47.4% | 70.8% | 89.1% |
| Margin | 0.931 | 48.2% | 70.8% | 89.1% |
| Entropy | 0.931 | 48.2% | 71.5% | 89.1% |
| Stability (8 flips/rotations) | **0.954** | **52.6%** | **81.0%** | 94.2% |
| TTA-averaged confidence | **0.954** | 49.6% | 77.4% | **94.9%** |

- **Stability is the strongest signal:** reviewing the least stable 10% catches 111 of 137 errors (vs. 97 for max probability, and about 14 for random review).
- **Accuracy without retraining:** averaging the 8 flipped/rotated predictions raises accuracy from 95.24% to **96.04%** (+0.80 points).
- **Confidence tracks correctness:** mean max probability is 0.979 for correct predictions vs. 0.775 for wrong ones; mean stability is 0.984 vs. 0.670.
- **Cigar-shaped galaxies (CSS) are flagged most:** 2% of the test set but 11.1% of the least-confident 10%, which contains 18 of their 23 errors.

## Note on the earlier preliminary run

The first run (`ds340w_novel_approach.ipynb`) evaluated the models on a split rebuilt from the sorted file list, because the authors' `read_split_data` picks images by `os.listdir` order and that order changed between Kaggle sessions. Many of those test images had been seen in training, so its numbers (97.01% accuracy, AUROC up to 0.970) were optimistic. The rerun fixes this by training and testing on the saved split in `../splits/`.
