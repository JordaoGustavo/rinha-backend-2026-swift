# Rinha Backend 2026 - Swift

Fraud detection API for [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026-q1), written in Swift 6.1.

## Architecture

- **SwiftNIO** raw HTTP/1.1 server (no framework overhead)
- **IVF Index** (Inverted File) with 4096 clusters over 3M 14-dim vectors
- **SIMD kernels** in C (AVX2 on x86_64, NEON on ARM64) for L2 distance
- **mmap'd binary indexes** — zero-copy, OS-managed paging
- **Int16 quantization** (scale=4096) with float32 reranking
- **SoA block layout** (8 vectors x 16 dims) for cache-efficient scanning
- **HAProxy** TCP mode with 2 API instances on Unix sockets

## Performance

Single instance (no HAProxy), Apple M-series:

| Metric | Value |
|--------|-------|
| Throughput (wrk, 50 conn) | ~114K RPS |
| p99 Latency (official k6 test) | 2.26ms |
| Official Rinha Score | **5645** |
| Detection Accuracy | 100% (0 FP, 0 FN) |

## Resource Constraints

```
Total: 1.0 CPU, 340 MB RAM
  HAProxy:    0.20 CPU, 20 MB
  API inst1:  0.40 CPU, 160 MB
  API inst2:  0.40 CPU, 160 MB
```

## Quick Start

```bash
# Download reference data
bash scripts/download-resources.sh

# Build and run with Docker
make docker-build
make docker-up

# Run official test
k6 run scripts/k6-official/test/test.js
```

## Build from Source

```bash
# Prerequisites: Swift 6.1+
swift build -c release

# Preprocess index (needs references.json.gz in data/)
.build/release/Preprocessor --input data/references.json.gz --output resources/

# Run server
.build/release/RinhaApp
```

## Project Structure

```
Sources/
  CSimd/              - C SIMD kernels (AVX2/NEON/scalar)
  FraudDetector/      - Core library (IVF search, parser, HTTP handler)
  Preprocessor/       - Index builder (K-means, IVF writer, exact writer)
  RinhaApp/           - Server entry point
docker/               - Dockerfile + docker-compose
lb/                   - HAProxy config
scripts/              - Benchmark and utility scripts
```

## License

MIT
