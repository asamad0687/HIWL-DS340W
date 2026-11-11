# Novel approach (in progress)

**Working title:** *Knowing When the Model Is Unsure: Confidence Scoring and Prediction Stability for Galaxy Morphology Classification*

This folder will hold the DS 340W novel-approach code. It's built **on top of** the reproduced HIWL pipeline in `../scheme/` and does not change the authors' training code.

Planned modules:

| Module | What it does | Uses |
|---|---|---|
| Confidence scoring | Gives each test galaxy a confidence score from HIWL's softmax output, so researchers can see which images the model is unsure about and by how much | Best model weights from `scheme/weights/B1/` |
| Prediction stability | Re-runs each galaxy with light flips and rotations and checks whether the predicted class changes (a second ambiguity signal) | Same weights + test split |
| Human review budget | How many of the model's errors are caught by reviewing only the least-confident X% of images | Outputs of the two modules above |
| Most-ambiguous figure | A grid of the ~10 least-confident / least-stable galaxies | Outputs above |
| Confusion-matrix view | Supporting visual of where errors cluster (e.g. cigar-shaped vs. edge-on) | `scheme/evaluate_metrics.py` |
