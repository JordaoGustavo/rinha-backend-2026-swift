#!/usr/bin/env python3
"""Train larger models to try to match k-NN accuracy."""
import gzip
import json
import time
import struct
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score

REFERENCES = "/Users/jordaogustavo/Documents/workspace/rinha_backend_2026/resources/references.json.gz"

print("=== Loading data ===")
t0 = time.time()
with gzip.open(REFERENCES, 'rt') as f:
    entries = json.load(f)

X = np.array([e['vector'] for e in entries], dtype=np.float32)
y = np.array([1 if e['label'] == 'fraud' else 0 for e in entries], dtype=np.int8)
print(f"Loaded {len(X)} vectors in {time.time()-t0:.1f}s")
print(f"Fraud: {y.sum()} ({y.mean()*100:.1f}%), Legit: {len(y)-y.sum()}")
print()

# Try Random Forest (better for complex boundaries)
print("=== Random Forest (500 trees, no max depth) ===")
t0 = time.time()
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,  # unlimited depth = can memorize
    min_samples_leaf=1,  # can memorize individual points
    n_jobs=-1,  # all cores
    random_state=42,
    verbose=1
)
rf.fit(X, y)
print(f"Training: {time.time()-t0:.1f}s")

y_pred = rf.predict(X)
acc = accuracy_score(y, y_pred)
errors = (y != y_pred).sum()
print(f"Train accuracy: {acc*100:.6f}% ({errors} errors)")
print()

# The real question: can ANY model besides k-NN get 100%?
# Let's check what the error pattern looks like
if errors > 0:
    wrong_idx = np.where(y != y_pred)[0]
    print(f"First 10 wrong predictions:")
    for i in wrong_idx[:10]:
        print(f"  idx={i}, true={y[i]}, pred={y_pred[i]}, vec={X[i][:5]}...")
print()
print(f"Conclusion: {errors} errors = {errors/len(y)*100:.4f}% error rate")
print(f"In rinha scoring: these errors would cost {errors * 3} weighted penalty points")
print(f"(FN=3 weight each)")
