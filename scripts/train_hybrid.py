#!/usr/bin/env python3
"""Train MLP for hybrid approach: ML fast-path + k-NN fallback."""
import gzip
import json
import time
import struct
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_predict

REFERENCES = "/Users/jordaogustavo/Documents/workspace/rinha_backend_2026/resources/references.json.gz"
OUTPUT = "/Users/jordaogustavo/Documents/workspace/rinha_backend_2026_swift/resources/model.bin"

print("=== Hybrid Model Training ===\n")

print("Loading references.json.gz...")
t0 = time.time()
with gzip.open(REFERENCES, 'rt') as f:
    entries = json.load(f)

X = np.array([e['vector'] for e in entries], dtype=np.float32)
y = np.array([1 if e['label'] == 'fraud' else 0 for e in entries], dtype=np.int8)
print(f"  {len(X)} vectors in {time.time()-t0:.1f}s")
print(f"  Fraud: {y.sum()} ({y.mean()*100:.1f}%)")
print()

# Train larger MLP
print("Training MLP (14 → 512 → 256 → 128 → 1)...")
t0 = time.time()
model = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128),
    activation='relu',
    solver='adam',
    max_iter=500,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    early_stopping=True,
    validation_fraction=0.02,
    n_iter_no_change=20,
    random_state=42,
    verbose=True,
    batch_size=4096
)
model.fit(X, y)
print(f"\nTraining: {time.time()-t0:.1f}s, {model.n_iter_} iterations")

# Get probabilities on training data
print("\nAnalyzing confidence distribution...")
probs = model.predict_proba(X)[:, 1]  # P(fraud)
preds = (probs >= 0.5).astype(int)
correct = (preds == y)

# Analyze by confidence threshold
print("\nThreshold analysis (confidence = |P - 0.5| * 2):")
print(f"{'Threshold':>10} {'ML handles':>12} {'ML errors':>10} {'→ kNN':>8} {'kNN catches':>12}")
print("-" * 65)

for thresh in [0.80, 0.85, 0.90, 0.95, 0.97, 0.99]:
    confidence = np.abs(probs - 0.5) * 2  # 0 = uncertain, 1 = very confident
    ml_mask = confidence >= thresh
    ml_count = ml_mask.sum()
    ml_errors = (~correct & ml_mask).sum()
    knn_count = (~ml_mask).sum()
    knn_catches = (~correct & ~ml_mask).sum()
    total_errors_if_no_fallback = (~correct).sum()

    print(f"{thresh:>10.2f} {ml_count:>10} ({ml_count/len(y)*100:>5.1f}%) "
          f"{ml_errors:>8} {knn_count:>8} ({knn_count/len(y)*100:>5.1f}%) "
          f"{knn_catches:>10}")

total_wrong = (~correct).sum()
print(f"\nTotal wrong without threshold: {total_wrong} ({total_wrong/len(y)*100:.4f}%)")

# Find the threshold where ML errors = 0
print("\nFinding safe threshold (0 ML errors)...")
confidence = np.abs(probs - 0.5) * 2
wrong_confidences = confidence[~correct]
if len(wrong_confidences) > 0:
    max_wrong_conf = wrong_confidences.max()
    safe_threshold = max_wrong_conf + 0.001
    ml_mask = confidence >= safe_threshold
    print(f"  Max confidence of wrong prediction: {max_wrong_conf:.6f}")
    print(f"  Safe threshold: {safe_threshold:.6f}")
    print(f"  ML handles: {ml_mask.sum()} ({ml_mask.sum()/len(y)*100:.1f}%)")
    print(f"  kNN handles: (~ml_mask).sum() ({(~ml_mask).sum()/len(y)*100:.1f}%)")
else:
    safe_threshold = 0.0
    print("  No wrong predictions! Threshold = 0")

# Export model
print(f"\nExporting model...")
weights = model.coefs_
biases = model.intercepts_

total_params = sum(w.size for w in weights) + sum(b.size for b in biases)
print(f"  Layers: {[w.shape for w in weights]}")
print(f"  Parameters: {total_params} ({total_params * 4 / 1024:.1f} KB)")

# Format: [num_layers(u32)][threshold_f32][for each: rows(u32), cols(u32), weights(f32...), bias(f32...)]
with open(OUTPUT, 'wb') as f:
    f.write(struct.pack('<I', len(weights)))
    f.write(struct.pack('<f', safe_threshold))
    for w, b in zip(weights, biases):
        rows, cols = w.shape
        f.write(struct.pack('<II', rows, cols))
        # Transpose weights for row-major access in Swift (input × weight)
        f.write(w.T.astype(np.float32).tobytes())
        f.write(b.astype(np.float32).tobytes())

import os
print(f"  Written to {OUTPUT} ({os.path.getsize(OUTPUT)/1024:.1f} KB)")
print(f"\n=== Summary ===")
print(f"  Model: 14 → 512 → 256 → 128 → 1")
print(f"  Size: {os.path.getsize(OUTPUT)/1024:.1f} KB")
print(f"  Safe threshold: {safe_threshold:.6f}")
ml_pct = ml_mask.sum()/len(y)*100 if len(wrong_confidences) > 0 else 100
print(f"  Fast-path (ML): ~{ml_pct:.0f}% of requests")
print(f"  Fallback (kNN): ~{100-ml_pct:.0f}% of requests")
