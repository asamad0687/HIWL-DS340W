"""
evaluate_metrics.py  (added for DS 340W reproduction)

Turns the predictions saved by train_efficientnet.py into the same metrics the
parent paper reports (Ma et al. 2023, MNRAS 519, Tables 3 and 4):
  - overall accuracy
  - per-class recall, precision, F1-score (+ arithmetic-mean "Avg" row)
  - 5x5 confusion matrix (rows = true class, columns = predicted class)

It does NOT retrain anything and does NOT change any of the authors' code.

Usage (run from the scheme/ folder, after train_efficientnet.py has finished):
    python evaluate_metrics.py --data-path5 /kaggle/working/clean_gzdata5 --num-model B1

No-retraining option (e.g. from a saved Kaggle version's output):
    python evaluate_metrics.py --pred-file <.../predicts/B1/test_pred_all.pth> \
                               --labels-csv <.../validation_unseen_set.csv>
  validation_unseen_set.csv is the authors' "test" split saved by the notebook
  (same order as test_pred_all.pth), so no data or GPU is needed.

Inputs:
  ./predicts/<num-model>/test_pred_all.pth   (written by train_efficientnet.py)
  --data-path5                                (same clean data folder used for training;
                                               the split is rebuilt with the authors'
                                               read_split_data, which uses random.seed(0))
"""
import argparse
import json
import os

import numpy as np
import torch

from utils import read_split_data

# Folder names -> class names used in the paper
FOLDER_TO_PAPER = {"bet": "IBS", "cig": "CSS", "edg": "EO", "rou": "CRS", "spi": "SPI"}

# Parent paper, Table 4 (HIWL on the test set) and overall accuracy
PAPER_TABLE4 = {
    "CRS": (0.9633, 0.9679, 0.9656),
    "IBS": (0.9628, 0.9499, 0.9563),
    "CSS": (0.7069, 0.7855, 0.7455),
    "EO":  (0.9616, 0.9641, 0.9629),
    "SPI": (0.9834, 0.9834, 0.9834),
    "Avg": (0.9156, 0.9302, 0.9227),
}
PAPER_OVERALL_ACC = 0.9632


def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_metrics(cm):
    tp = np.diag(cm).astype(float)
    recall = np.divide(tp, cm.sum(axis=1), out=np.zeros_like(tp), where=cm.sum(axis=1) > 0)
    precision = np.divide(tp, cm.sum(axis=0), out=np.zeros_like(tp), where=cm.sum(axis=0) > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return recall, precision, f1


def main(args):
    pred_path = args.pred_file or os.path.join("predicts", args.num_model, "test_pred_all.pth")
    assert os.path.exists(pred_path), f"{pred_path} not found - run train_efficientnet.py first."
    y_pred = torch.load(pred_path, map_location="cpu").long().numpy()

    if args.labels_csv:
        # Labels saved by the notebook (authors' "test" split, same order as the predictions)
        import csv
        with open(args.labels_csv) as f:
            y_true = np.array([int(r["label"]) for r in csv.DictReader(f)], dtype=int)
    else:
        assert args.data_path5, "Give either --labels-csv or --data-path5."
        # Rebuild the exact same split the training script used (seeded inside read_split_data)
        _, _, _, _, _, test_labels = read_split_data(args.data_path5)
        y_true = np.array(test_labels, dtype=int)
    assert len(y_true) == len(y_pred), (
        f"Label count ({len(y_true)}) != prediction count ({len(y_pred)}). "
        "Labels and predictions must come from the same run.")

    # Class index -> folder name (sorted folder names, as in utils.read_split_data)
    idx_to_folder = {"0": "bet", "1": "cig", "2": "edg", "3": "rou", "4": "spi"}
    if os.path.exists("class_indices.json"):
        with open("class_indices.json") as f:
            idx_to_folder = json.load(f)
    names = [FOLDER_TO_PAPER.get(idx_to_folder[str(i)], idx_to_folder[str(i)])
             for i in range(len(idx_to_folder))]

    cm = confusion_matrix(y_true, y_pred, len(names))
    recall, precision, f1 = per_class_metrics(cm)
    acc = np.trace(cm) / cm.sum()

    # Print in the paper's row order
    order = [names.index(n) for n in ["CRS", "IBS", "CSS", "EO", "SPI"] if n in names]

    print("\n=== Overall accuracy (test set) ===")
    print(f"Ours: {acc * 100:.2f}%   Paper: {PAPER_OVERALL_ACC * 100:.2f}%")

    print("\n=== Per-class metrics vs. parent paper Table 4 ===")
    print(f"{'Class':<6}{'Recall':>10}{'(paper)':>9}{'Precision':>11}{'(paper)':>9}{'F1':>8}{'(paper)':>9}")
    rows = []
    for i in order:
        n = names[i]
        pr, pp, pf = PAPER_TABLE4[n]
        print(f"{n:<6}{recall[i]:>10.4f}{pr:>9.4f}{precision[i]:>11.4f}{pp:>9.4f}{f1[i]:>8.4f}{pf:>9.4f}")
        rows.append((n, recall[i], precision[i], f1[i]))
    pr, pp, pf = PAPER_TABLE4["Avg"]
    print(f"{'Avg':<6}{recall.mean():>10.4f}{pr:>9.4f}{precision.mean():>11.4f}{pp:>9.4f}{f1.mean():>8.4f}{pf:>9.4f}")

    print("\n=== Confusion matrix (rows = true, cols = predicted) ===")
    header = "".join(f"{names[j]:>7}" for j in order)
    print(f"{'':<6}{header}")
    for i in order:
        print(f"{names[i]:<6}" + "".join(f"{cm[i, j]:>7}" for j in order))

    # Save a CSV for the README / paper
    os.makedirs("results", exist_ok=True)
    out_csv = os.path.join("results", f"metrics_{args.num_model}.csv")
    with open(out_csv, "w") as f:
        f.write("class,recall,precision,f1,paper_recall,paper_precision,paper_f1\n")
        for n, r, p, f1v in rows:
            f.write(f"{n},{r:.4f},{p:.4f},{f1v:.4f},{','.join(f'{v:.4f}' for v in PAPER_TABLE4[n])}\n")
        f.write(f"Avg,{recall.mean():.4f},{precision.mean():.4f},{f1.mean():.4f},"
                f"{','.join(f'{v:.4f}' for v in PAPER_TABLE4['Avg'])}\n")
        f.write(f"overall_accuracy,{acc:.4f},,,{PAPER_OVERALL_ACC:.4f},,\n")
    print(f"\nSaved {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path5", type=str, default=None)
    parser.add_argument("--num-model", default="B1")
    parser.add_argument("--pred-file", type=str, default=None,
                        help="path to test_pred_all.pth (default: ./predicts/<num-model>/test_pred_all.pth)")
    parser.add_argument("--labels-csv", type=str, default=None,
                        help="validation_unseen_set.csv saved by the notebook (skips rebuilding the split)")
    main(parser.parse_args())
