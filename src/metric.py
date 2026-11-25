"""Official challenge metric (AN2DL 2026-27): weighted F1 over the 10 classes.

Per-class F1, averaged with weights n_c / N (test support of each class).
Single shared implementation: use it everywhere (CV, ensembling, report).
"""
import numpy as np
from sklearn.metrics import f1_score

N_CLASSES = 10
AVERAGE = 'weighted'


def score(y_true, y_pred):
    """y_true, y_pred: 1D integer labels in [0, 9]. Returns weighted F1."""
    return f1_score(np.asarray(y_true), np.asarray(y_pred),
                    labels=np.arange(N_CLASSES), average=AVERAGE, zero_division=0)


def score_from_logits(y_true, logits):
    """Convenience wrapper for (N, C) logits or probabilities."""
    return score(y_true, np.argmax(np.asarray(logits), axis=1))


if __name__ == '__main__':
    # Unit test on a known case
    y_t = np.array([0, 0, 0, 1, 1, 2])
    y_p = np.array([0, 0, 1, 1, 1, 2])
    # F1: class0 = 0.8 (w=3/6), class1 = 0.8 (w=2/6), class2 = 1.0 (w=1/6) -> 0.8333...
    assert abs(score(y_t, y_p) - (0.8 * 3 + 0.8 * 2 + 1.0) / 6) < 1e-9, score(y_t, y_p)
    print('metric.py OK:', score(y_t, y_p))
