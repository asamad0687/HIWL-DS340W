"""
hiwl_uncertainty.py  (DS 340W novel approach, Abdul Samad)

"Knowing When the Model Is Unsure: Confidence Scoring and Prediction Stability
for Galaxy Morphology Classification"

Adds an uncertainty layer on top of the reproduced HIWL model (Ma et al. 2023).
It does NOT retrain or change the authors' model. It loads the two saved
EfficientNet-B1 models (best_model4_0.pth, best_model2_0.pth), runs them on the
same seeded test split, and for every test galaxy computes:

  1. Max probability (MSP)   - probability of the predicted class
  2. Margin                  - predicted-class probability minus the best other class
  3. Entropy                 - how spread out the 5-class probabilities are
  4. Stability               - share of 8 flipped/rotated views that keep the same prediction
  5. TTA confidence          - MSP after averaging the 8 views' probabilities

HIWL is hierarchical, so the 5-class probabilities are built from both models:
  P(bet) = p4[0], P(rou) = p4[2], P(spi) = p4[3]
  P(cig) = p4[1] * p2[0],  P(edg) = p4[1] * p2[1]
(model4 merges cigar + edge-on into one group; model2 splits that group.)
The predicted class follows the authors' rule in utils.evaluateall exactly.

Outputs (in --out-dir):
  per_image_scores.csv, comparison_table.csv, review_budget.csv,
  flagged_by_class.csv, summary.json, most_ambiguous.png

Usage (from the repo's scheme/ folder):
  python ../novel/hiwl_uncertainty.py --data-path5 <clean_gzdata5> \
      --weights-dir <.../weights/B1> --ref-pred <.../predicts/B1/test_pred_all.pth>
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scheme"))
from model_efficientnet import efficientnet_b1  # noqa: E402  (authors' code)
from my_dataset import MyDataSet                # noqa: E402  (authors' code)
from utils import read_split_data               # noqa: E402  (authors' code)

CLASS_NAMES = ["IBS", "CSS", "EO", "CRS", "SPI"]  # index 0..4 = bet, cig, edg, rou, spi
CLASSES4 = [0, 1, 3, 4]  # model4 output index -> 5-class label (authors' evaluateall)
CLASSES2 = [1, 2]        # model2 output index -> 5-class label
BUDGETS = [0.05, 0.10, 0.20]
PARENT_ACC = 0.9632      # Ma et al. (2023), HIWL test accuracy


# --------------------------------------------------------------------------- #
# 8 exact flips/rotations (the dihedral group). Galaxies have no "up", so the
# true class never changes; the model's answer shouldn't either.
# --------------------------------------------------------------------------- #
def views(x):
    r = [torch.rot90(x, k, dims=(2, 3)) for k in range(4)]
    return r + [torch.flip(v, dims=(3,)) for v in r]


def hierarchical_predict(p4, p2):
    """Authors' rule: model4 picks a group; if it is the cigar/edge-on group,
    model2 decides which of the two."""
    g = p4.argmax(1)
    pred = torch.tensor(CLASSES4, device=p4.device)[g]
    sub = torch.tensor(CLASSES2, device=p4.device)[p2.argmax(1)]
    return torch.where(g == 1, sub, pred)


def five_class_probs(p4, p2):
    return torch.stack([p4[:, 0], p4[:, 1] * p2[:, 0], p4[:, 1] * p2[:, 1],
                        p4[:, 2], p4[:, 3]], dim=1)


def load(num_classes, path, device):
    m = efficientnet_b1(num_classes=num_classes).to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    return m.eval()


@torch.no_grad()
def run(model4, model2, loader, device):
    out = {k: [] for k in ["label", "pred", "p5", "pred_views", "p5_tta", "pred_tta"]}
    for images, labels in loader:
        images = images.to(device)
        p4s, p2s, preds = [], [], []
        for v in views(images):
            p4 = F.softmax(model4(v), 1)
            p2 = F.softmax(model2(v), 1)
            p4s.append(p4); p2s.append(p2); preds.append(hierarchical_predict(p4, p2))
        p4m, p2m = torch.stack(p4s).mean(0), torch.stack(p2s).mean(0)
        out["label"].append(labels)
        out["pred"].append(preds[0].cpu())                    # view 0 = original image
        out["p5"].append(five_class_probs(p4s[0], p2s[0]).cpu())
        out["pred_views"].append(torch.stack(preds, 1).cpu())
        out["p5_tta"].append(five_class_probs(p4m, p2m).cpu())
        out["pred_tta"].append(hierarchical_predict(p4m, p2m).cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


# --------------------------------------------------------------------------- #
# Scoring. Every score is an "uncertainty": higher = less sure = review first.
# --------------------------------------------------------------------------- #
def uncertainty_scores(r):
    n = len(r["pred"])
    idx = np.arange(n)
    p5 = r["p5"]
    msp = p5[idx, r["pred"]]
    others = p5.copy(); others[idx, r["pred"]] = -1
    margin = msp - others.max(1)
    q = p5 / p5.sum(1, keepdims=True)
    entropy = -(q * np.log(np.clip(q, 1e-12, 1))).sum(1)
    stability = (r["pred_views"] == r["pred"][:, None]).mean(1)
    tta_msp = r["p5_tta"][idx, r["pred_tta"]]
    return {
        "Max probability (MSP)": -msp,
        "Margin (top-1 minus top-2)": -margin,
        "Entropy": entropy,
        # stability has only 9 possible values; MSP breaks ties
        "Stability (8 flips/rotations)": -(stability + 1e-3 * msp),
        "TTA-averaged confidence": -tta_msp,
    }, dict(msp=msp, margin=margin, entropy=entropy, stability=stability, tta_msp=tta_msp)


def auroc(score, is_err):
    """Chance that a random wrong prediction is ranked as more uncertain than a
    random correct one (Mann-Whitney U, ties counted half)."""
    from scipy.stats import rankdata
    ranks = rankdata(score)
    n1, n0 = is_err.sum(), (~is_err).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[is_err].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def budget_stats(score, is_err, frac):
    n = len(score)
    k = int(round(frac * n))
    order = np.argsort(-score, kind="stable")
    flagged = np.zeros(n, bool); flagged[order[:k]] = True
    caught = int((flagged & is_err).sum())
    rest = ~flagged
    return dict(k=k, caught=caught, pct_caught=caught / max(is_err.sum(), 1),
                acc_after_review=1 - (is_err & rest).sum() / n,   # humans fix the flagged ones
                acc_unflagged=1 - is_err[rest].mean())


def main(a):
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out_dir, exist_ok=True)

    _, _, _, _, test_paths, test_labels = read_split_data(a.data_path5)
    tf = transforms.Compose([transforms.CenterCrop(240), transforms.ToTensor(),
                             transforms.Normalize([0.046, 0.041, 0.030], [0.090, 0.075, 0.065])])
    if a.image_root:  # split decided on --data-path5, images read from here
        test_paths = [p.replace(a.data_path5, a.image_root) for p in test_paths]
    ds = MyDataSet(test_paths, test_labels, transform=tf)
    loader = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                                         num_workers=a.workers, collate_fn=MyDataSet.collate_fn)

    model4 = load(4, os.path.join(a.weights_dir, "best_model4_0.pth"), device)
    model2 = load(2, os.path.join(a.weights_dir, "best_model2_0.pth"), device)
    r = run(model4, model2, loader, device)

    y, pred = r["label"], r["pred"]
    is_err = pred != y
    acc = 1 - is_err.mean()
    acc_tta = (r["pred_tta"] == y).mean()
    summary = dict(n_test=int(len(y)), errors=int(is_err.sum()), accuracy=float(acc),
                   accuracy_tta=float(acc_tta), parent_paper_accuracy=PARENT_ACC)
    print(f"\nTest images: {len(y)}   errors: {is_err.sum()}")
    print(f"HIWL accuracy (reproduced): {acc:.4f}   TTA-averaged accuracy: {acc_tta:.4f}")

    # Sanity check: same images, same models -> same predictions as the reproduction run
    if a.ref_pred and os.path.exists(a.ref_pred):
        ref = torch.load(a.ref_pred, map_location="cpu").long().numpy()
        agree = float((ref == pred).mean()) if len(ref) == len(pred) else 0.0
        summary["agreement_with_reproduction_run"] = agree
        print(f"Agreement with saved reproduction predictions: {agree:.4f}")
        if agree < 0.995:
            print("WARNING: predictions differ from the reproduction run - the test split "
                  "may not be the same images. Do not trust these results yet.")

    scores, raw = uncertainty_scores(r)

    rows, budget_rows = [], []
    for name, s in scores.items():
        row = dict(method=name, auroc_error_detection=auroc(s, is_err))
        for b in BUDGETS:
            st = budget_stats(s, is_err, b)
            row[f"errors_caught_{int(b*100)}pct"] = st["pct_caught"]
            budget_rows.append(dict(method=name, review_budget=b, images_reviewed=st["k"],
                                    errors_caught=st["caught"], pct_errors_caught=st["pct_caught"],
                                    accuracy_after_review=st["acc_after_review"],
                                    accuracy_of_unflagged=st["acc_unflagged"]))
        rows.append(row)
    rand = dict(method="Random review (baseline)", auroc_error_detection=0.5)
    for b in BUDGETS:
        rand[f"errors_caught_{int(b*100)}pct"] = b
    rows.insert(0, rand)

    import pandas as pd
    comp = pd.DataFrame(rows)
    comp.to_csv(os.path.join(a.out_dir, "comparison_table.csv"), index=False)
    pd.DataFrame(budget_rows).to_csv(os.path.join(a.out_dir, "review_budget.csv"), index=False)
    print("\n=== Comparison of uncertainty methods (higher = better) ===")
    print(comp.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # Which classes end up in the 10% review slice (MSP), vs. their share of the test set
    k = int(round(0.10 * len(y)))
    flag = np.argsort(-scores["Max probability (MSP)"], kind="stable")[:k]
    fb = pd.DataFrame({
        "class": CLASS_NAMES,
        "test_share": [float((y == c).mean()) for c in range(5)],
        "share_of_flagged_10pct": [float((y[flag] == c).mean()) for c in range(5)],
        "errors": [int(((y == c) & is_err).sum()) for c in range(5)],
        "errors_flagged_10pct": [int(((y[flag] == c) & is_err[flag]).sum()) for c in range(5)],
    })
    fb.to_csv(os.path.join(a.out_dir, "flagged_by_class.csv"), index=False)
    print("\n=== Classes in the least-confident 10% (MSP) ===")
    print(fb.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    per = pd.DataFrame(dict(image=[os.path.basename(p) for p in test_paths],
                            true=[CLASS_NAMES[i] for i in y], pred=[CLASS_NAMES[i] for i in pred],
                            correct=~is_err, pred_tta=[CLASS_NAMES[i] for i in r["pred_tta"]], **raw))
    per.to_csv(os.path.join(a.out_dir, "per_image_scores.csv"), index=False)

    # Mean confidence of right vs wrong predictions
    summary["mean_msp_correct"] = float(raw["msp"][~is_err].mean())
    summary["mean_msp_wrong"] = float(raw["msp"][is_err].mean())
    summary["mean_stability_correct"] = float(raw["stability"][~is_err].mean())
    summary["mean_stability_wrong"] = float(raw["stability"][is_err].mean())
    summary["share_fully_stable"] = float((raw["stability"] == 1).mean())
    with open(os.path.join(a.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + json.dumps(summary, indent=2))

    # Figure: the 10 least-confident galaxies
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
        worst = np.argsort(raw["msp"])[:10]
        fig, axes = plt.subplots(2, 5, figsize=(15, 7.8))
        for ax, i in zip(axes.flat, worst):
            ax.imshow(Image.open(test_paths[i]).convert("RGB").crop((92, 92, 332, 332)))
            ax.set_title(f"true {CLASS_NAMES[y[i]]} / pred {CLASS_NAMES[pred[i]]}\n"
                         f"conf {raw['msp'][i]:.2f}, stable {raw['stability'][i]:.2f}",
                         fontsize=10, color="black" if y[i] == pred[i] else "crimson")
            ax.axis("off")
        fig.suptitle("The 10 test galaxies HIWL is least sure about (red = wrong prediction)")
        fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.5)
        fig.savefig(os.path.join(a.out_dir, "most_ambiguous.png"), dpi=130)
        print("Saved most_ambiguous.png")
    except Exception as e:  # figure is optional
        print("Figure skipped:", e)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-path5", required=True)
    p.add_argument("--weights-dir", required=True)
    p.add_argument("--ref-pred", default=None, help="predicts/B1/test_pred_all.pth from the reproduction")
    p.add_argument("--out-dir", default="results/novel")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--image-root", default=None,
                   help="read images from here instead (same file names as --data-path5)")
    p.add_argument("--device", default="cuda:0")
    main(p.parse_args())
