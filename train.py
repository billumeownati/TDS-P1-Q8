#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pandas", "scipy"]
# ///
"""Train the linear-rank-v1 face-expression model and write model.json.

    uv run train.py data/features.csv [more-features.csv ...] --output model.json

Every CSV must have the course layout: person_id, photo_id, intended_expression, then 78 features.
Only people who have exactly one photo for each of the ten expressions are used.
A person who appears in several CSVs (same numbers) is counted once.
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment, minimize
from scipy.special import log_softmax, softmax
from scipy.stats import rankdata

LABELS = ["neutral", "happy", "sad", "angry", "surprised", "contempt",
          "restrained-amusement", "scepticism", "uncertainty", "hopelessness"]


def load(paths):
    """Return (X, feature_names). X has shape (people, 10, 78); photo k of each person shows LABELS[k]."""
    people, names, seen, dupes = [], None, set(), 0
    for n, path in enumerate(paths):
        d = pd.read_csv(path, dtype={"person_id": str, "photo_id": str})
        cols = list(d.columns[3:])
        if len(cols) != 78 or (names and cols != names):
            raise SystemExit(f"{path}: expected the same 78 feature columns as the course CSV")
        names = cols
        for person, g in d.groupby("person_id"):
            if sorted(g.intended_expression) != sorted(LABELS):
                print(f"skip {path}:{person} (needs exactly one photo per expression)")
                continue
            g = g.set_index("intended_expression").loc[LABELS]
            x = g[cols].to_numpy(float)
            if x.tobytes() in seen:                   # same person already loaded from another CSV
                dupes += 1
                continue
            seen.add(x.tobytes())
            people.append(x)
    if dupes:
        print(f"dropped {dupes} people who appear in more than one CSV")
    return np.stack(people), names


def mirror_index(names):
    """Column order that turns each face into its mirror image (swap ...Left with ...Right)."""
    idx = np.arange(len(names))
    for i, name in enumerate(names):
        if name.endswith("Left"):
            j = names.index(name[:-4] + "Right")
            idx[i], idx[j] = j, i
    return idx


def ranks(X):
    """Each feature's rank among that person's photos, 0..1, ties averaged - same as the grader."""
    return (rankdata(X, axis=1) - 1) / (X.shape[1] - 1)


def inputs(X):
    """156 training inputs. Subtracting a per-person constant is free: it cannot change the matching."""
    return np.concatenate([X - X.mean(1, keepdims=True), ranks(X) - 0.5], axis=2)


def fit(X, alpha, swap):
    """Fit on people X. Returns a 10 x 156 matrix that applies to [raw features, ranks]."""
    X = np.concatenate([X, X[:, :, swap]])            # add every face's mirror image
    Z = inputs(X)
    scale = Z.reshape(-1, Z.shape[2]).std(0) + 1e-12  # like StandardScaler.scale_
    Z = Z / scale
    P, _, D = Z.shape
    eye = np.eye(10)

    def loss(w):
        W = w.reshape(10, D)
        S = Z @ W.T                                   # S[person, photo, expression]
        # "which expression is this photo?" + "which photo shows this expression?"
        value = -(np.einsum("pjj->", log_softmax(S, 2)) + np.einsum("pjj->", log_softmax(S, 1))) / P
        G = (softmax(S, 2) - eye) + (softmax(S, 1) - eye)
        grad = np.einsum("pjk,pjd->kd", G, Z) / P
        return value + alpha * (W ** 2).sum(), (grad + 2 * alpha * W).ravel()

    W = minimize(loss, np.zeros(10 * D), jac=True, method="L-BFGS-B", options={"maxiter": 500}).x
    return W.reshape(10, D) / scale                   # undo the scaling: coefficient / scale_


def grade(W, X):
    """Do what the grader does: score, then match each expression to exactly one photo."""
    hits = 0
    for person in X:
        scores = np.hstack([person, ranks(person[None])[0]]) @ np.asarray(W).T   # photo x expression
        expr, photo = linear_sum_assignment(-scores.T)
        hits += (expr == photo).sum()                 # photo k really shows LABELS[k]
    return hits / (10 * len(X))


def cross_validate(X, alpha, swap, folds=6, repeats=5, train_people=None):
    """Accuracy on people the model never saw, averaged over several random splits."""
    accs = []
    for seed in range(repeats):
        rng = np.random.RandomState(seed)
        fold = rng.permutation(len(X)) % folds
        hits = 0
        for k in range(folds):
            train = np.where(fold != k)[0]
            if train_people:
                train = rng.choice(train, train_people, replace=False)
            hits += grade(fit(X[train], alpha, swap), X[fold == k]) * (fold == k).sum()
        accs.append(hits / len(X))
    return float(np.mean(accs)), float(np.std(accs))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--output", default="model.json")
    ap.add_argument("--alphas", type=float, nargs="+", default=[1, 3, 10], help="regularisation strengths to try")
    ap.add_argument("--learning-curve", action="store_true", help="also show accuracy vs. number of training people")
    args = ap.parse_args()

    X, names = load(args.csv)
    swap = mirror_index(names)
    print(f"{len(X)} people x 10 photos x {X.shape[2]} features")

    results = {}
    for alpha in args.alphas:
        results[alpha] = cross_validate(X, alpha, swap)
        print(f"alpha={alpha:<6g} accuracy on unseen people = {results[alpha][0]:.1%} (+-{results[alpha][1]:.1%})")
    best = max(results, key=lambda a: results[a][0])
    print(f"best alpha = {best:g}")

    if args.learning_curve:
        max_train = len(X) - -(-len(X) // 6)          # people left after holding out the largest fold
        for n in sorted({n for n in (5, 10, 15, 20, 40, 80, 160) if n < max_train} | {max_train}):
            acc, _ = cross_validate(X, best, swap, train_people=n)
            print(f"trained on {n:3d} people -> {acc:.1%}")

    W = fit(X, best, swap)
    model = {"version": "linear-rank-v1", "weights": [[float(f"{w:.6g}") for w in row] for row in W]}
    text = json.dumps(model, separators=(",", ":"))
    with open(args.output, "w") as f:
        f.write(text)

    # Re-read the file and check it the way the grader will.
    saved = json.loads(open(args.output).read())
    Wj = np.array(saved["weights"])
    assert set(saved) == {"version", "weights"} and Wj.shape == (10, 156) and np.isfinite(Wj).all()
    assert len(text.encode()) <= 64 * 1024
    print(f"wrote {args.output}: {len(text.encode())} bytes, accuracy on its own training people = {grade(Wj, X):.1%}")


if __name__ == "__main__":
    main()
