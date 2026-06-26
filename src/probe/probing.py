"""Day 5 — linear probes on FROZEN V-JEPA 2 latents.

A linear probe answers: is property P linearly decodable from the latents?
High accuracy => the world model *represents* P. (It does NOT prove the model
*uses* P for prediction — note that caveat in your writeup.)
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def collect_features(model, loader, label_fn, device="cuda"):
    """Return X (frozen embeddings) and y (labels for property P)."""
    feats, labels = [], []
    # TODO(Day5): for each batch -> frozen model embeddings -> X; label_fn(batch) -> y
    raise NotImplementedError
    return np.array(feats), np.array(labels)


def probe(X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    acc = clf.score(Xte, yte)
    print(f"linear-probe accuracy: {acc:.3f}")
    return acc


if __name__ == "__main__":
    # TODO(Day5): choose property P (e.g., motion direction / object presence),
    # define label_fn, run probe, record accuracy in README results table.
    pass
