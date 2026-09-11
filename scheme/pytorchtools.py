import numpy as np
import torch

class EarlyStopping:
    """Stops training if validation accuracy doesn't improve after a given patience."""
    def __init__(self, patience=25, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = 0
        self.delta = delta

    def __call__(self, val_loss, val_acc, model):
        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.val_acc_max = val_acc
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_acc_max = val_acc
            self.counter = 0

