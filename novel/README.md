# Novel approach: confidence scoring and prediction stability

**Working title:** *Knowing When the Model Is Unsure: Confidence Scoring and Prediction Stability for Galaxy Morphology Classification*

This folder holds the DS 340W novel-approach code. It's built **on top of** the reproduced HIWL pipeline in `../scheme/`. It doesn't retrain the model or change the authors' code. It only changes the **inference / evaluation block**: instead of keeping just HIWL's final class, it keeps the model's probabilities and uses them to flag the galaxies the model is unsure about.

| File | What it is |
|---|---|
| `hiwl_uncertainty.py` | The novel-approach script (loads the saved HIWL models, scores every test galaxy, writes the results) |
| `ds340w_novel_approach.ipynb` | The Kaggle notebook that runs it (Kaggle: *DS340W Novel Approach*, GPU T4 x2, about 2 minutes) |
| `results/` | Output tables from the Kaggle run |

## How it works

HIWL is hierarchical. **Model 4** sorts a galaxy into 4 groups (in-between smooth, round, spiral, or "cigar **or** edge-on"), and **Model 2** splits that last group into cigar vs. edge-on. The script combines both models into one 5-class probability per galaxy:

- P(IBS), P(CRS), P(SPI) come from Model 4
- P(CSS) = P4(cigar-or-edge-on) × P2(cigar), and P(EO) = P4(cigar-or-edge-on) × P2(edge-on)

The predicted class follows the authors' rule in `utils.evaluateall` exactly (checked against it).

**Five ways to flag uncertain predictions** (the comparison):

| Method | Idea |
|---|---|
| Max probability (MSP) | How high is the predicted class's probability? |
| Margin | How far ahead is the top class of the second one? |
| Entropy | How spread out are the probabilities across all 5 classes? |
| Stability | Re-run the galaxy with 8 exact flips/rotations. How often does the answer stay the same? |
| TTA-averaged confidence | MSP after averaging the 8 flipped/rotated predictions |

**Human review budget:** sort galaxies from least to most confident, send only the bottom 5%, 10% or 20% to a human, and count how many of the model's mistakes that catches.

## How to run (Kaggle)

1. New notebook → add the **DS340W** reproduction notebook's output as Input (it has `weights/B1/best_model4_0.pth`, `best_model2_0.pth` and `clean_gzdata5`).
2. GPU T4 x2 on → import `ds340w_novel_approach.ipynb` → Run All.

Or directly, from `scheme/`:
```bash
python ../novel/hiwl_uncertainty.py --data-path5 <clean_gzdata5> --weights-dir <.../weights/B1> \
    --ref-pred <.../predicts/B1/test_pred_all.pth> --out-dir results_novel
```

## Preliminary results (2,881 test galaxies, 86 errors)

| Method | AUROC (error detection) | Errors caught, review 5% | Review 10% | Review 20% |
|---|---|---|---|---|
| Random review (baseline) | 0.500 | 5.0% | 10.0% | 20.0% |
| Max probability (MSP) | 0.963 | 73.3% | 88.4% | 94.2% |
| Margin | 0.963 | 73.3% | 88.4% | 94.2% |
| Entropy | 0.963 | **74.4%** | 88.4% | 94.2% |
| Stability (8 flips/rotations) | **0.970** | 67.4% | **91.9%** | 97.7% |
| TTA-averaged confidence | 0.958 | 54.7% | 88.4% | **98.8%** |

- **Accuracy:** HIWL 97.01% → **97.40%** when the 8 flipped/rotated predictions are averaged (+0.39 points, no retraining).
- **Confidence tracks correctness:** mean MSP is 0.979 for correct predictions vs. 0.700 for wrong ones; mean stability is 0.982 vs. 0.593.
- **Cigar-shaped galaxies (CSS) are flagged most:** they're 2% of the test set but 12.5% of the least-confident 10%, and that slice catches 17 of their 19 errors.

### Important caveat: the test split

The authors' `read_split_data` picks the test images from the order `os.listdir` returns. The reproduction ran on a working-disk folder whose order depends on that session's disk, so it **can't be rebuilt** in a new session. (Rebuilding the folder on `/kaggle/working` gave 95% agreement, not 100%.)

Here the split is rebuilt from the read-only Input folder, which lists files in sorted order, so **anyone re-running this gets the same split**. But it's a different 10% than the reproduction's test set (94.0% prediction agreement with the reproduction run), so many of these images were seen during training. That's why accuracy is 97.0% rather than the reproduction's 96.49%.

All methods are compared on the same images, so the **ranking between methods** is a fair comparison. The **absolute numbers are preliminary and likely optimistic**. Planned fix: retrain HIWL once with this fixed, saved split.
