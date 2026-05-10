#!/usr/bin/env python3
"""Train a small MLP to replace k-NN for fraud detection."""
import gzip
import json
import time
import struct
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

REFERENCES = "/Users/jordaogustavo/Documents/workspace/rinha_backend_2026/resources/references.json.gz"
TEST_DATA = "/Users/jordaogustavo/Documents/workspace/rinha_backend_2026/scripts/k6-official/test/test-data.json"
OUTPUT = "/Users/jordaogustavo/Documents/workspace/rinha_backend_2026_swift/resources/model.bin"

print("=== Training fraud detection MLP ===\n")

# Load references
print("Loading references.json.gz...")
t0 = time.time()
with gzip.open(REFERENCES, 'rt') as f:
    entries = json.load(f)

X_train = np.array([e['vector'] for e in entries], dtype=np.float32)
y_train = np.array([1 if e['label'] == 'fraud' else 0 for e in entries], dtype=np.int8)
print(f"  Loaded {len(X_train)} vectors in {time.time()-t0:.1f}s")
print(f"  Fraud: {y_train.sum()}, Legit: {len(y_train)-y_train.sum()}")
print(f"  Dims: {X_train.shape[1]}")
print()

# Train MLP
print("Training MLP (14 → 256 → 128 → 64 → 1)...")
t0 = time.time()
model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    solver='adam',
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.05,
    random_state=42,
    verbose=True
)
model.fit(X_train, y_train)
print(f"  Training completed in {time.time()-t0:.1f}s")
print(f"  Iterations: {model.n_iter_}")
print()

# Evaluate on training data
print("Evaluating on training data...")
y_pred_train = model.predict(X_train)
train_acc = accuracy_score(y_train, y_pred_train)
print(f"  Train accuracy: {train_acc*100:.4f}%")
train_errors = (y_train != y_pred_train).sum()
print(f"  Train errors: {train_errors}")
print()

# Load and evaluate on rinha test data
print("Loading rinha test data...")
with open(TEST_DATA, 'r') as f:
    test_data = json.load(f)

test_entries = test_data['entries']
print(f"  {len(test_entries)} test entries")

# We need to transform test entries the same way the API does
# The test entries have 'request' (the API payload) and 'expected_approved'
# We need the TransactionParser logic to convert request → 14-dim vector
# For now, let's just validate on training data accuracy

# Export model weights
print("\nExporting model weights...")
weights = model.coefs_  # list of weight matrices
biases = model.intercepts_  # list of bias vectors

total_params = sum(w.size for w in weights) + sum(b.size for b in biases)
print(f"  Layers: {[w.shape for w in weights]}")
print(f"  Total parameters: {total_params}")
print(f"  Size: {total_params * 4 / 1024:.1f} KB (float32)")

# Write binary format: [num_layers(u32)][for each: rows(u32), cols(u32), weights(f32...), bias(f32...)]
with open(OUTPUT, 'wb') as f:
    f.write(struct.pack('<I', len(weights)))  # num layers
    for w, b in zip(weights, biases):
        rows, cols = w.shape
        f.write(struct.pack('<II', rows, cols))
        f.write(w.astype(np.float32).tobytes())
        f.write(b.astype(np.float32).tobytes())

import os
file_size = os.path.getsize(OUTPUT)
print(f"  Written to {OUTPUT} ({file_size/1024:.1f} KB)")
print()
print(f"=== Done! Train accuracy: {train_acc*100:.4f}% ({train_errors} errors on {len(y_train)} samples) ===")
