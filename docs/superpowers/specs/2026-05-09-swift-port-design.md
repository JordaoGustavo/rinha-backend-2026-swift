# Rinha Backend 2026 — Swift Port Design

## Overview

Port the rinha_backend_2026 fraud-detection API from C#/.NET 11 to Swift 6 using raw SwiftNIO. The system classifies financial transactions as fraudulent or approved using an IVF (Inverted File) index with k-nearest neighbor search over 3M reference vectors.

## Constraints

- **Same API contract**: `POST /fraud-score`, `GET /ready`
- **Same binary index formats**: `ivf.bin` (v7), `exact.bin` (v1)
- **Same resource budget**: 1.0 CPU core, 340 MB total (HAProxy 0.20/16MB, 2x API 0.40/162MB)
- **Same HAProxy config**: TCP mode, round-robin, Unix domain sockets

## Architecture

```
Client → HAProxy (TCP :9999, round-robin, Unix sockets)
         ├→ Swift API 1 (cpuset 1,2, 162 MB, 0.4 cores)
         └→ Swift API 2 (cpuset 2,3, 162 MB, 0.4 cores)
```

### Components

1. **HttpHandler** — Raw SwiftNIO `ChannelHandler` for HTTP/1.1 on Unix sockets
2. **TransactionParser** — Manual UTF-8 byte scanner, zero Foundation dependency
3. **IvfDetector** — mmap'd IVF index, SIMD int16 distance, centroid search, bbox repair
4. **ExactDetector** — mmap'd float32 vectors for reranking top-6 → top-5
5. **SimdDistance** — C-bridged SIMD kernels (AVX2/NEON) + Swift scalar fallback
6. **Preprocessor** — K-means clustering (K=4096), IVF/exact binary generation

## Design Decisions

### HTTP Layer: Raw SwiftNIO

SwiftNIO `ServerBootstrap` → Unix domain socket. Custom `ChannelInboundHandler` parses raw HTTP/1.1 bytes. No Vapor/Hummingbird overhead. Pre-built 12 cached UTF-8 response buffers (approved/denied × fraud_count 0-5). Matches the C# `SocketHttpServer` approach.

Key details:
- Bind to Unix socket path from `SOCKET_PATH` env var
- `SO_REUSEADDR`, TCP keepalive
- Body accumulation in single `ByteBuffer`, reused across requests
- Route dispatch: first byte check (`G` = GET /ready, `P` = POST /fraud-score)
- No HTTP parsing library — manual header skip to body

### JSON Parsing: Manual UTF-8 Scanner

Foundation's `JSONDecoder` allocates heavily. Instead: scan raw `UnsafeRawBufferPointer` byte-by-byte. Field dispatch by key hash or prefix matching. Parse numbers inline (no String intermediary). ISO-8601 timestamp: digit-by-digit extraction matching C# approach. FNV-1a hash for merchant ID matching in known_merchants array.

### SIMD: C Bridge + Swift Fallback

Swift's `simd` module handles 2/4/8-element vectors but lacks 16-wide int16 operations needed for IVF distance. Solution:

- **CSimd C module**: AVX2 (`__m256i`) and NEON (`int16x8_t`) kernels for:
  - `int16_l2_squared(a, b, dims=16)` — int16 L2 distance
  - `int16_bbox_lower_bound(query, bmin, bmax, dims=16)` — bbox gap distance
  - `float32_l2_squared(a, b, dims=16)` — float32 L2 for rerank
- **Runtime dispatch**: compile-time `#if arch(x86_64)` / `#if arch(arm64)` for platform selection
- **Swift scalar fallback**: pure Swift implementation for testing/debugging

### Memory: mmap + UnsafeRawPointer

```swift
let fd = open(path, O_RDONLY)
let ptr = mmap(nil, size, PROT_READ, MAP_PRIVATE, fd, 0)
// Prefault: walk every 4KB page
// madvise MADV_HUGEPAGE (Linux), MADV_RANDOM (exact.bin)
```

Access via `UnsafeRawPointer.load(fromByteOffset:as:)` for headers, `UnsafeBufferPointer<Int16>` for vectors. No copying — read directly from mapped memory.

### Feature Vector (14 dimensions, same as C#)

```
[0]  amount / 10000
[1]  installments / 12
[2]  (amount / customer.avg_amount) / 10, clamped 0-1
[3]  hour / 23
[4]  day_of_week / 6 (Zeller's congruence)
[5]  minutes_since_last_tx / 1440 (or -1 if null)
[6]  km_from_current / 1000 (or -1 if null)
[7]  km_from_home / 1000
[8]  tx_count_24h / 20
[9]  is_online ? 1.0 : 0.0
[10] card_present ? 1.0 : 0.0
[11] is_unknown_merchant ? 1.0 : 0.0 (FNV-1a hash)
[12] mcc_risk[mcc] (lookup table, default 0.5)
[13] merchant.avg_amount / 10000
```

### IVF Search Algorithm (identical to C#)

1. Quantize query to int16 (scale=4096)
2. Int16 L2 to all 4096 centroids → top-5 nearest (max-heap)
3. Scan selected clusters: AoSoA blocks of 8, early-exit partial sums, prefetch
4. Bbox repair pass: bitset for scanned clusters, bbox lower-bound skip
5. Top-6 int16 candidates
6. Float32 rerank via exact.bin → top-5 final
7. Decision: fraud_count = sum(labels[top5]), approved = (fraud_count < 3)

### Preprocessor

Full port of the C# IVF builder:
1. Download `references.json.gz` from official rinha URL
2. Parse 3M vectors + labels from NDJSON
3. K-means clustering: K=4096, 20 iterations, random init
4. Build IVF binary: centroids, bbox, cluster metadata, reordered vectors, labels, original indices
5. Build exact binary: float32 vectors + labels
6. Write `ivf.bin` (magic "IVFR", v7) and `exact.bin` (magic "EXCT", v1)

### Docker

Multi-stage build:
1. **Build stage** (`swift:6.1`): download data, run preprocessor, build release binary with `-c release -Xswiftc -O -Xswiftc -whole-module-optimization`
2. **Runtime stage** (`ubuntu:24.04` or `swift:6.1-slim`): copy binary + resources + index files
3. Static linking where possible to minimize runtime dependencies

## Project Structure

```
rinha_backend_2026_swift/
├── Package.swift
├── Sources/
│   ├── RinhaApp/                  # Main executable
│   │   └── main.swift
│   ├── FraudDetector/             # Core detection library
│   │   ├── HttpHandler.swift      # SwiftNIO HTTP handler
│   │   ├── IvfDetector.swift      # IVF search engine
│   │   ├── ExactDetector.swift    # Float32 reranker
│   │   ├── TransactionParser.swift # Zero-alloc JSON parser
│   │   ├── FeatureVector.swift    # 14-dim vector type
│   │   ├── ResponseCache.swift    # Pre-built JSON responses
│   │   └── MccRisk.swift          # MCC risk lookup table
│   ├── Preprocessor/              # Index building executable
│   │   ├── PreprocessorMain.swift
│   │   ├── KMeans.swift
│   │   ├── IvfBuilder.swift
│   │   ├── IvfBinaryWriter.swift
│   │   ├── ExactBinaryWriter.swift
│   │   └── DataLoader.swift
│   └── CSimd/                     # C SIMD kernels
│       ├── include/
│       │   └── simd_distance.h
│       └── simd_distance.c
├── Tests/
│   └── FraudDetectorTests/
│       ├── TransactionParserTests.swift
│       ├── IvfDetectorTests.swift
│       └── SimdDistanceTests.swift
├── docker/
│   └── Dockerfile
├── docker-compose.yml
├── lb/
│   └── haproxy.cfg
├── resources/
│   ├── mcc_risk.json
│   └── normalization.json
├── scripts/
│   ├── download_data.sh
│   └── gen_warmup.sh
├── Makefile
├── .gitignore
├── .dockerignore
└── README.md
```

## API Contract

### GET /ready
- Returns `200 OK` once index is mmap'd, prefaulted, and warmup queries completed
- Empty body

### POST /fraud-score
- Input: JSON transaction (same schema as C# version)
- Output: `{"approved": true/false, "fraud_score": 0.0-1.0}`
- Never returns 5xx — fallback to `{"approved": true, "fraud_score": 0.0}` on any error

## Testing Strategy

1. **Unit tests**: TransactionParser (golden payload round-trip, Zeller's, timestamp delta)
2. **Unit tests**: SimdDistance (int16 L2, float32 L2, bbox lower bound — compare C vs Swift)
3. **Integration tests**: IvfDetector with small test index
4. **Docker tests**: Full stack with HAProxy, k6 load test
5. **Accuracy test**: IVF vs brute-force exact oracle on 10k queries

## Risk Mitigations

| Risk | Mitigation |
|------|-----------|
| Swift SIMD slower than .NET AVX2 | C bridge with platform-specific intrinsics |
| Swift ARC overhead vs C# value types | Use `UnsafeBufferPointer`, avoid class allocations in hot path |
| JSONDecoder too slow | Manual UTF-8 byte scanner, no Foundation in hot path |
| Docker image too large | Static linking, minimal base image |
| Memory budget exceeded | Profile early, tune buffer sizes, same mmap strategy |
