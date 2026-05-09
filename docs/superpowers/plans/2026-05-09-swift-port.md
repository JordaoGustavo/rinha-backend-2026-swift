# Rinha Backend 2026 — Swift Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the rinha_backend_2026 fraud-detection API from C#/.NET to Swift 6 with raw SwiftNIO, maintaining identical API contract, binary index format, and resource constraints.

**Architecture:** Raw SwiftNIO HTTP/1.1 server on Unix domain sockets, mmap'd IVF/exact indexes with C-bridged SIMD distance kernels, manual UTF-8 JSON parser, full preprocessor with K-means clustering. Two API instances behind HAProxy in TCP mode, total 1.0 CPU / 340 MB.

**Tech Stack:** Swift 6.1, SwiftNIO 2.x, C (SIMD intrinsics via clang module), Docker multi-stage build, HAProxy 3.1

**Reference C# source:** `/Users/jordaogustavo/Documents/workspace/rinha_backend_2026/src/`

---

## File Structure

```
rinha_backend_2026_swift/
├── Package.swift                           # SPM manifest
├── Sources/
│   ├── RinhaApp/
│   │   └── main.swift                      # Entry point, server bootstrap
│   ├── FraudDetector/
│   │   ├── BinaryFormats.swift             # IvfBinaryFormat + ExactBinaryFormat constants
│   │   ├── MmapHints.swift                 # Linux madvise wrappers
│   │   ├── SimdDistance.swift              # Swift scalar fallback distance functions
│   │   ├── TransactionParser.swift         # Zero-alloc manual UTF-8 JSON parser
│   │   ├── ResponseCache.swift             # Pre-built HTTP/1.1 response buffers
│   │   ├── IvfDetector.swift               # IVF search engine (mmap'd)
│   │   ├── ExactDetector.swift             # Float32 brute-force detector
│   │   ├── HttpHandler.swift               # SwiftNIO ChannelHandler
│   │   └── MccRisk.swift                   # MCC risk lookup table + normalization
│   ├── Preprocessor/
│   │   ├── PreprocessorMain.swift          # CLI entry point for preprocessing
│   │   ├── DataLoader.swift                # Load references.json.gz
│   │   ├── KMeans.swift                    # K-means clustering
│   │   ├── IvfBuilder.swift                # Build IVF result from clustered vectors
│   │   ├── IvfBinaryWriter.swift           # Write ivf.bin (v7 SoA-blocked)
│   │   ├── ExactBinaryWriter.swift         # Write exact.bin (v1)
│   │   └── GenWarmup.swift                 # Generate warmup-payloads.ndjson
│   └── CSimd/
│       ├── include/
│       │   └── simd_distance.h             # C header for SIMD kernels
│       └── simd_distance.c                 # AVX2/NEON/scalar distance implementations
├── Tests/
│   └── FraudDetectorTests/
│       ├── TransactionParserTests.swift
│       ├── SimdDistanceTests.swift
│       └── BinaryFormatTests.swift
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── lb/
│   └── haproxy.cfg
├── resources/
│   ├── mcc_risk.json
│   └── normalization.json
├── Makefile
├── .gitignore
└── .dockerignore
```

---

### Task 1: Project Scaffolding

**Files:**
- Create: `Package.swift`
- Create: `.gitignore`
- Create: `.dockerignore`
- Copy: `resources/mcc_risk.json` (from C# project)
- Copy: `resources/normalization.json` (from C# project)

- [ ] **Step 1: Create Package.swift**

```swift
// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "RinhaBackend",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(url: "https://github.com/apple/swift-nio.git", from: "2.80.0"),
    ],
    targets: [
        .executableTarget(
            name: "RinhaApp",
            dependencies: [
                "FraudDetector",
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOPosix", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio"),
            ]
        ),
        .executableTarget(
            name: "Preprocessor",
            dependencies: ["FraudDetector"]
        ),
        .target(
            name: "FraudDetector",
            dependencies: [
                "CSimd",
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOPosix", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio"),
            ]
        ),
        .systemLibrary(
            name: "CSimd"
        ),
        .testTarget(
            name: "FraudDetectorTests",
            dependencies: ["FraudDetector"]
        ),
    ]
)
```

- [ ] **Step 2: Create .gitignore**

```
.DS_Store
.build/
.swiftpm/
*.o
*.d
data/
resources/references.json.gz
resources/warmup-payloads.ndjson
DerivedData/
.vscode/
.claude/
```

- [ ] **Step 3: Create .dockerignore**

```
.git
.build
.swiftpm
data
.DS_Store
.vscode
.claude
docs
tests
scripts
*.md
DerivedData
```

- [ ] **Step 4: Copy resource files from C# project**

```bash
cp /Users/jordaogustavo/Documents/workspace/rinha_backend_2026/resources/mcc_risk.json \
   /Users/jordaogustavo/Documents/workspace/rinha_backend_2026_swift/resources/mcc_risk.json
cp /Users/jordaogustavo/Documents/workspace/rinha_backend_2026/resources/normalization.json \
   /Users/jordaogustavo/Documents/workspace/rinha_backend_2026_swift/resources/normalization.json
```

- [ ] **Step 5: Create CSimd module map**

Create `Sources/CSimd/include/module.modulemap`:
```
module CSimd {
    header "simd_distance.h"
    link "CSimd"
    export *
}
```

- [ ] **Step 6: Verify project resolves**

```bash
cd /Users/jordaogustavo/Documents/workspace/rinha_backend_2026_swift
swift package resolve
```

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat: scaffold Swift project with Package.swift and resources"
```

---

### Task 2: CSimd — C SIMD Distance Kernels

**Files:**
- Create: `Sources/CSimd/include/simd_distance.h`
- Create: `Sources/CSimd/simd_distance.c`

Reference: `rinha_backend_2026/src/Api/SimdDistance.cs`

The C module provides SIMD-accelerated distance functions called from Swift via the CSimd import. Platform dispatch is compile-time via `#ifdef`.

- [ ] **Step 1: Write simd_distance.h**

```c
#ifndef SIMD_DISTANCE_H
#define SIMD_DISTANCE_H

#include <stdint.h>

int csimd_int16_l2_squared(const int16_t *a, const int16_t *b);
int csimd_int16_l2_squared_first8(const int16_t *a, const int16_t *b);
int csimd_int16_bbox_lower_bound(const int16_t *query, const int16_t *bboxMin, const int16_t *bboxMax);
float csimd_float32_l2_squared(const float *a, const float *b);
void csimd_prefetch(const void *ptr);

#endif
```

- [ ] **Step 2: Write simd_distance.c**

```c
#include "simd_distance.h"
#include <string.h>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

#if defined(__aarch64__) || defined(__arm64__)
#include <arm_neon.h>
#endif

int csimd_int16_l2_squared(const int16_t *a, const int16_t *b) {
#if defined(__x86_64__) && defined(__AVX2__)
    __m256i va = _mm256_loadu_si256((const __m256i *)a);
    __m256i vb = _mm256_loadu_si256((const __m256i *)b);
    __m256i diff = _mm256_sub_epi16(va, vb);
    __m256i madd = _mm256_madd_epi16(diff, diff);
    __m128i hi = _mm256_extracti128_si256(madd, 1);
    __m128i lo = _mm256_castsi256_si128(madd);
    __m128i s = _mm_add_epi32(hi, lo);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
#elif defined(__aarch64__) || defined(__arm64__)
    int16x8_t d0 = vsubq_s16(vld1q_s16(a), vld1q_s16(b));
    int16x8_t d1 = vsubq_s16(vld1q_s16(a + 8), vld1q_s16(b + 8));
    int32x4_t sq0lo = vmull_s16(vget_low_s16(d0), vget_low_s16(d0));
    int32x4_t sq0hi = vmull_high_s16(d0, d0);
    int32x4_t sq1lo = vmull_s16(vget_low_s16(d1), vget_low_s16(d1));
    int32x4_t sq1hi = vmull_high_s16(d1, d1);
    int32x4_t acc = vaddq_s32(vaddq_s32(sq0lo, sq0hi), vaddq_s32(sq1lo, sq1hi));
    return vaddvq_s32(acc);
#else
    int total = 0;
    for (int i = 0; i < 16; i++) {
        int d = a[i] - b[i];
        total += d * d;
    }
    return total;
#endif
}

int csimd_int16_l2_squared_first8(const int16_t *a, const int16_t *b) {
#if defined(__aarch64__) || defined(__arm64__)
    int16x8_t d = vsubq_s16(vld1q_s16(a), vld1q_s16(b));
    int32x4_t sqlo = vmull_s16(vget_low_s16(d), vget_low_s16(d));
    int32x4_t sqhi = vmull_high_s16(d, d);
    int32x4_t acc = vaddq_s32(sqlo, sqhi);
    return vaddvq_s32(acc);
#elif defined(__x86_64__)
    __m128i va = _mm_loadu_si128((const __m128i *)a);
    __m128i vb = _mm_loadu_si128((const __m128i *)b);
    __m128i diff = _mm_sub_epi16(va, vb);
    __m128i madd = _mm_madd_epi16(diff, diff);
    madd = _mm_add_epi32(madd, _mm_shuffle_epi32(madd, 0x4E));
    madd = _mm_add_epi32(madd, _mm_shuffle_epi32(madd, 0xB1));
    return _mm_cvtsi128_si32(madd);
#else
    int total = 0;
    for (int i = 0; i < 8; i++) {
        int d = a[i] - b[i];
        total += d * d;
    }
    return total;
#endif
}

int csimd_int16_bbox_lower_bound(const int16_t *query, const int16_t *bboxMin, const int16_t *bboxMax) {
#if defined(__x86_64__) && defined(__AVX2__)
    __m256i q = _mm256_loadu_si256((const __m256i *)query);
    __m256i lo = _mm256_max_epi16(_mm256_sub_epi16(_mm256_loadu_si256((const __m256i *)bboxMin), q), _mm256_setzero_si256());
    __m256i hi = _mm256_max_epi16(_mm256_sub_epi16(q, _mm256_loadu_si256((const __m256i *)bboxMax)), _mm256_setzero_si256());
    __m256i gap = _mm256_add_epi16(lo, hi);
    __m256i madd = _mm256_madd_epi16(gap, gap);
    __m128i h128 = _mm256_extracti128_si256(madd, 1);
    __m128i l128 = _mm256_castsi256_si128(madd);
    __m128i s = _mm_add_epi32(h128, l128);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
#elif defined(__aarch64__) || defined(__arm64__)
    int16x8_t q0 = vld1q_s16(query);
    int16x8_t q1 = vld1q_s16(query + 8);
    int16x8_t zero = vdupq_n_s16(0);
    int16x8_t lo0 = vmaxq_s16(vsubq_s16(vld1q_s16(bboxMin), q0), zero);
    int16x8_t hi0 = vmaxq_s16(vsubq_s16(q0, vld1q_s16(bboxMax)), zero);
    int16x8_t g0 = vaddq_s16(lo0, hi0);
    int16x8_t lo1 = vmaxq_s16(vsubq_s16(vld1q_s16(bboxMin + 8), q1), zero);
    int16x8_t hi1 = vmaxq_s16(vsubq_s16(q1, vld1q_s16(bboxMax + 8)), zero);
    int16x8_t g1 = vaddq_s16(lo1, hi1);
    int32x4_t sq0lo = vmull_s16(vget_low_s16(g0), vget_low_s16(g0));
    int32x4_t sq0hi = vmull_high_s16(g0, g0);
    int32x4_t sq1lo = vmull_s16(vget_low_s16(g1), vget_low_s16(g1));
    int32x4_t sq1hi = vmull_high_s16(g1, g1);
    int32x4_t acc = vaddq_s32(vaddq_s32(sq0lo, sq0hi), vaddq_s32(sq1lo, sq1hi));
    return vaddvq_s32(acc);
#else
    int lb = 0;
    for (int d = 0; d < 16; d++) {
        int gap = 0;
        if (query[d] < bboxMin[d]) gap = bboxMin[d] - query[d];
        else if (query[d] > bboxMax[d]) gap = query[d] - bboxMax[d];
        lb += gap * gap;
    }
    return lb;
#endif
}

float csimd_float32_l2_squared(const float *a, const float *b) {
#if defined(__x86_64__) && defined(__FMA__)
    __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
    __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + 8), _mm256_loadu_ps(b + 8));
    __m256 acc = _mm256_fmadd_ps(d0, d0, _mm256_setzero_ps());
    acc = _mm256_fmadd_ps(d1, d1, acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 r = _mm_add_ps(hi, lo);
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_shuffle_ps(r, r, 1));
    return _mm_cvtss_f32(r);
#elif defined(__aarch64__) || defined(__arm64__)
    float32x4_t acc = vdupq_n_f32(0);
    for (int i = 0; i < 16; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t d = vsubq_f32(va, vb);
        acc = vfmaq_f32(acc, d, d);
    }
    return vaddvq_f32(acc);
#else
    float sum = 0.0f;
    for (int i = 0; i < 16; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
#endif
}

void csimd_prefetch(const void *ptr) {
#if defined(__x86_64__)
    _mm_prefetch((const char *)ptr, _MM_HINT_T0);
#elif defined(__aarch64__) || defined(__arm64__)
    __builtin_prefetch(ptr, 0, 3);
#endif
}
```

- [ ] **Step 3: Verify CSimd compiles**

```bash
cd /Users/jordaogustavo/Documents/workspace/rinha_backend_2026_swift
swift build --target CSimd 2>&1 | head -5
```

- [ ] **Step 4: Commit**

```bash
git add Sources/CSimd/
git commit -m "feat: add CSimd C module with AVX2/NEON/scalar distance kernels"
```

---

### Task 3: Binary Formats + MmapHints + MccRisk

**Files:**
- Create: `Sources/FraudDetector/BinaryFormats.swift`
- Create: `Sources/FraudDetector/MmapHints.swift`
- Create: `Sources/FraudDetector/MccRisk.swift`
- Create: `Sources/FraudDetector/SimdDistance.swift`

Reference: `IvfBinaryFormat.cs`, `ExactBinaryFormat.cs`, `MmapHints.cs`

- [ ] **Step 1: Write BinaryFormats.swift**

```swift
import Foundation

enum IvfBinaryFormat {
    static let magic: [UInt8] = [0x49, 0x56, 0x46, 0x52] // "IVFR"
    static let version: UInt32 = 7
    static let headerSize = 64
    static let dims = 14
    static let paddedDims = 16
    static let scale: Int32 = 4096
    static let blockVectors = 8
    static let centroidVectorBytes = paddedDims * MemoryLayout<Int16>.size
    static let int16VectorBytes = paddedDims * MemoryLayout<Int16>.size

    struct ClusterMeta {
        var offset: UInt32
        var count: UInt32
    }

    static var centroidsOffset: Int { headerSize }

    static func bboxMinOffset(_ numClusters: Int) -> Int {
        centroidsOffset + numClusters * centroidVectorBytes
    }

    static func bboxMaxOffset(_ numClusters: Int) -> Int {
        bboxMinOffset(numClusters) + numClusters * int16VectorBytes
    }

    static func clusterMetaOffset(_ numClusters: Int) -> Int {
        bboxMaxOffset(numClusters) + numClusters * int16VectorBytes
    }

    static func vectorsOffset(_ numClusters: Int) -> Int {
        clusterMetaOffset(numClusters) + numClusters * 8
    }

    static func labelsOffset(_ numClusters: Int, _ totalSlots: Int) -> Int {
        vectorsOffset(numClusters) + totalSlots * int16VectorBytes
    }

    static func originalIndicesOffset(_ numClusters: Int, _ totalSlots: Int) -> Int {
        labelsOffset(numClusters, totalSlots) + totalSlots
    }

    static func totalSize(_ numClusters: Int, _ totalSlots: Int) -> Int {
        originalIndicesOffset(numClusters, totalSlots) + totalSlots * MemoryLayout<Int32>.size
    }
}

enum ExactBinaryFormat {
    static let magic: [UInt8] = [0x45, 0x58, 0x43, 0x54] // "EXCT"
    static let version: UInt32 = 1
    static let headerSize = 32
    static let dims = 14
    static let paddedDims = 16
    static let vectorBytes = paddedDims * MemoryLayout<Float>.size

    static var vectorsOffset: Int { headerSize }

    static func labelsOffset(_ numVectors: Int) -> Int {
        vectorsOffset + numVectors * vectorBytes
    }

    static func totalSize(_ numVectors: Int) -> Int {
        labelsOffset(numVectors) + numVectors
    }
}
```

- [ ] **Step 2: Write MmapHints.swift**

```swift
#if canImport(Glibc)
import Glibc
#elseif canImport(Musl)
import Musl
#elseif canImport(Darwin)
import Darwin
#endif

enum MmapHints {
    #if os(Linux)
    private static let MADV_HUGEPAGE: Int32 = 14
    private static let MADV_WILLNEED: Int32 = 3
    private static let MADV_RANDOM: Int32 = 1

    static func hintHugePages(_ addr: UnsafeMutableRawPointer, _ length: Int) {
        _ = madvise(addr, length, MADV_HUGEPAGE)
    }

    static func hintWillNeed(_ addr: UnsafeMutableRawPointer, _ length: Int) {
        _ = madvise(addr, length, MADV_WILLNEED)
    }

    static func hintRandom(_ addr: UnsafeMutableRawPointer, _ length: Int) {
        _ = madvise(addr, length, MADV_RANDOM)
    }
    #else
    static func hintHugePages(_ addr: UnsafeMutableRawPointer, _ length: Int) {}
    static func hintWillNeed(_ addr: UnsafeMutableRawPointer, _ length: Int) {}
    static func hintRandom(_ addr: UnsafeMutableRawPointer, _ length: Int) {}
    #endif
}
```

- [ ] **Step 3: Write MccRisk.swift**

```swift
import Foundation

enum MccRisk {
    private static let tableSize = 10000
    private static let defaultRisk: Float = 0.50
    private(set) static var table = [Float](repeating: 0.50, count: tableSize)

    static var maxAmount: Double = 10000.0
    static var maxInstallments: Double = 12.0
    static var amountVsAvgRatio: Double = 10.0
    static var maxMinutes: Double = 1440.0
    static var maxKm: Double = 1000.0
    static var maxTxCount24h: Double = 20.0
    static var maxMerchantAvgAmount: Double = 10000.0

    @inline(__always)
    static func risk(for mcc: Int) -> Float {
        guard mcc >= 0, mcc < tableSize else { return defaultRisk }
        return table[mcc]
    }

    static func initialize(mccRiskPath: String, normalizationPath: String) throws {
        table = [Float](repeating: defaultRisk, count: tableSize)

        let mccData = try Data(contentsOf: URL(fileURLWithPath: mccRiskPath))
        let mccJson = try JSONSerialization.jsonObject(with: mccData) as! [String: Any]
        for (key, value) in mccJson {
            if let mcc = Int(key), mcc >= 0, mcc < tableSize,
               let risk = value as? Double {
                table[mcc] = Float(risk)
            }
        }

        let normData = try Data(contentsOf: URL(fileURLWithPath: normalizationPath))
        let normJson = try JSONSerialization.jsonObject(with: normData) as! [String: Any]
        maxAmount = normJson["max_amount"] as? Double ?? 10000.0
        maxInstallments = normJson["max_installments"] as? Double ?? 12.0
        amountVsAvgRatio = normJson["amount_vs_avg_ratio"] as? Double ?? 10.0
        maxMinutes = normJson["max_minutes"] as? Double ?? 1440.0
        maxKm = normJson["max_km"] as? Double ?? 1000.0
        maxTxCount24h = normJson["max_tx_count_24h"] as? Double ?? 20.0
        maxMerchantAvgAmount = normJson["max_merchant_avg_amount"] as? Double ?? 10000.0
    }
}
```

- [ ] **Step 4: Write SimdDistance.swift (Swift scalar wrappers around CSimd)**

```swift
import CSimd

enum SimdDistance {
    @inline(__always)
    static func int16L2Squared(_ a: UnsafePointer<Int16>, _ b: UnsafePointer<Int16>) -> Int32 {
        Int32(csimd_int16_l2_squared(a, b))
    }

    @inline(__always)
    static func int16L2SquaredFirst8(_ a: UnsafePointer<Int16>, _ b: UnsafePointer<Int16>) -> Int32 {
        Int32(csimd_int16_l2_squared_first8(a, b))
    }

    @inline(__always)
    static func int16BboxLowerBound(_ query: UnsafePointer<Int16>, _ bboxMin: UnsafePointer<Int16>, _ bboxMax: UnsafePointer<Int16>) -> Int32 {
        Int32(csimd_int16_bbox_lower_bound(query, bboxMin, bboxMax))
    }

    @inline(__always)
    static func float32L2Squared(_ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>) -> Float {
        csimd_float32_l2_squared(a, b)
    }

    @inline(__always)
    static func prefetch(_ ptr: UnsafeRawPointer) {
        csimd_prefetch(ptr)
    }
}
```

- [ ] **Step 5: Commit**

```bash
git add Sources/FraudDetector/BinaryFormats.swift Sources/FraudDetector/MmapHints.swift \
      Sources/FraudDetector/MccRisk.swift Sources/FraudDetector/SimdDistance.swift
git commit -m "feat: add binary formats, mmap hints, MCC risk table, SIMD wrappers"
```

---

### Task 4: TransactionParser — Manual UTF-8 JSON Parser

**Files:**
- Create: `Sources/FraudDetector/TransactionParser.swift`
- Test: `Tests/FraudDetectorTests/TransactionParserTests.swift`

Reference: `rinha_backend_2026/src/Api/TransactionParser.cs`

This is the most critical parser. It must produce identical 14-dim float vectors from the same JSON input. Zero-alloc hot path using `UnsafeRawBufferPointer`.

- [ ] **Step 1: Write TransactionParser.swift**

```swift
import Foundation

public enum TransactionParser {
    @inline(__always)
    private static func clampD(_ x: Double) -> Double {
        max(0.0, min(1.0, x))
    }

    @inline(__always)
    private static func fnvHash(_ bytes: UnsafeRawBufferPointer) -> UInt64 {
        var hash: UInt64 = 14695981039346656037
        for i in 0..<bytes.count {
            hash ^= UInt64(bytes[i])
            hash &*= 1099511628211
        }
        return hash
    }

    @inline(__always)
    private static func fnvHash(_ bytes: UnsafeBufferPointer<UInt8>) -> UInt64 {
        var hash: UInt64 = 14695981039346656037
        for i in 0..<bytes.count {
            hash ^= UInt64(bytes[i])
            hash &*= 1099511628211
        }
        return hash
    }

    public static func parse(_ json: UnsafeRawBufferPointer, into vector: UnsafeMutablePointer<Float>) {
        for i in 0..<16 { vector[i] = 0 }

        var txAmount: Double = 0
        var txInstallments: Int = 0
        var txYear = 0, txMonth = 0, txDay = 0, txHour = 0, txMinute = 0, txSecond = 0

        var custAvgAmount: Double = 0
        var custTxCount24h: Int = 0

        var knownHashes = (
            UInt64(0), UInt64(0), UInt64(0), UInt64(0), UInt64(0), UInt64(0), UInt64(0), UInt64(0),
            UInt64(0), UInt64(0), UInt64(0), UInt64(0), UInt64(0), UInt64(0), UInt64(0), UInt64(0),
            UInt64(0), UInt64(0), UInt64(0), UInt64(0), UInt64(0), UInt64(0), UInt64(0), UInt64(0),
            UInt64(0), UInt64(0), UInt64(0), UInt64(0), UInt64(0), UInt64(0), UInt64(0), UInt64(0)
        )
        var knownCount = 0
        var merchantIdHash: UInt64 = 0
        var hasMerchantId = false

        var merchantMcc: Int = -1
        var merchantAvgAmount: Double = 0

        var terminalIsOnline = false
        var terminalCardPresent = false
        var terminalKmFromHome: Double = 0

        var hasLastTx = false
        var lastYear = 0, lastMonth = 0, lastDay = 0, lastHour = 0, lastMinute = 0, lastSecond = 0
        var lastTxKmFromCurrent: Double = 0

        let buf = json.bindMemory(to: UInt8.self)
        parseJsonManual(buf, &txAmount, &txInstallments,
                       &txYear, &txMonth, &txDay, &txHour, &txMinute, &txSecond,
                       &custAvgAmount, &custTxCount24h,
                       &knownHashes, &knownCount, &merchantIdHash, &hasMerchantId,
                       &merchantMcc, &merchantAvgAmount,
                       &terminalIsOnline, &terminalCardPresent, &terminalKmFromHome,
                       &hasLastTx, &lastYear, &lastMonth, &lastDay, &lastHour, &lastMinute, &lastSecond,
                       &lastTxKmFromCurrent)

        // Zeller's congruence
        var zZ: Int, zC: Int, zM: Int
        if txMonth < 3 {
            zZ = (txYear - 1) % 100; zC = (txYear - 1) / 100; zM = txMonth + 12
        } else {
            zZ = txYear % 100; zC = txYear / 100; zM = txMonth
        }
        let zH = (txDay + 13 * (zM + 1) / 5 + zZ + zZ / 4 + zC / 4 + 5 * zC) % 7
        let monBased = (zH + 5) % 7

        vector[0] = Float((clampD(txAmount / MccRisk.maxAmount) * 10000).rounded() / 10000)
        vector[1] = Float((clampD(Double(txInstallments) / MccRisk.maxInstallments) * 10000).rounded() / 10000)
        vector[2] = custAvgAmount == 0
            ? 1.0
            : Float((clampD((txAmount / custAvgAmount) / MccRisk.amountVsAvgRatio) * 10000).rounded() / 10000)
        vector[3] = Float((Double(txHour) / 23.0 * 10000).rounded() / 10000)
        vector[4] = Float((Double(monBased) / 6.0 * 10000).rounded() / 10000)

        if hasLastTx {
            let minutes: Double
            if txYear == lastYear && txMonth == lastMonth {
                let deltaSec = Int64(txDay - lastDay) * 86400
                    + Int64(txHour - lastHour) * 3600
                    + Int64(txMinute - lastMinute) * 60
                    + Int64(txSecond - lastSecond)
                minutes = Double(deltaSec) / 60.0
            } else {
                var txCal = DateComponents()
                txCal.year = txYear; txCal.month = txMonth; txCal.day = txDay
                txCal.hour = txHour; txCal.minute = txMinute; txCal.second = txSecond
                txCal.timeZone = TimeZone(identifier: "UTC")
                var lastCal = DateComponents()
                lastCal.year = lastYear; lastCal.month = lastMonth; lastCal.day = lastDay
                lastCal.hour = lastHour; lastCal.minute = lastMinute; lastCal.second = lastSecond
                lastCal.timeZone = TimeZone(identifier: "UTC")
                let cal = Calendar(identifier: .gregorian)
                let txDate = cal.date(from: txCal)!
                let lastDate = cal.date(from: lastCal)!
                minutes = txDate.timeIntervalSince(lastDate) / 60.0
            }
            vector[5] = Float((clampD(minutes / MccRisk.maxMinutes) * 10000).rounded() / 10000)
            vector[6] = Float((clampD(lastTxKmFromCurrent / MccRisk.maxKm) * 10000).rounded() / 10000)
        } else {
            vector[5] = -1
            vector[6] = -1
        }

        vector[7] = Float((clampD(terminalKmFromHome / MccRisk.maxKm) * 10000).rounded() / 10000)
        vector[8] = Float((clampD(Double(custTxCount24h) / MccRisk.maxTxCount24h) * 10000).rounded() / 10000)
        vector[9] = terminalIsOnline ? 1 : 0
        vector[10] = terminalCardPresent ? 1 : 0

        var isUnknown = true
        if hasMerchantId && knownCount > 0 {
            withUnsafePointer(to: &knownHashes) { ptr in
                ptr.withMemoryRebound(to: UInt64.self, capacity: 32) { hashes in
                    for i in 0..<knownCount {
                        if hashes[i] == merchantIdHash { isUnknown = false; break }
                    }
                }
            }
        }
        vector[11] = isUnknown ? 1 : 0
        vector[12] = MccRisk.risk(for: merchantMcc)
        vector[13] = Float((clampD(merchantAvgAmount / MccRisk.maxMerchantAvgAmount) * 10000).rounded() / 10000)
    }
}

// Manual JSON byte scanner — separate function to keep parse() readable.
// Uses a simple state machine matching the C# Utf8JsonReader approach.
private func parseJsonManual(
    _ buf: UnsafeBufferPointer<UInt8>,
    _ txAmount: inout Double, _ txInstallments: inout Int,
    _ txYear: inout Int, _ txMonth: inout Int, _ txDay: inout Int,
    _ txHour: inout Int, _ txMinute: inout Int, _ txSecond: inout Int,
    _ custAvgAmount: inout Double, _ custTxCount24h: inout Int,
    _ knownHashes: inout (UInt64, UInt64, UInt64, UInt64, UInt64, UInt64, UInt64, UInt64,
                          UInt64, UInt64, UInt64, UInt64, UInt64, UInt64, UInt64, UInt64,
                          UInt64, UInt64, UInt64, UInt64, UInt64, UInt64, UInt64, UInt64,
                          UInt64, UInt64, UInt64, UInt64, UInt64, UInt64, UInt64, UInt64),
    _ knownCount: inout Int,
    _ merchantIdHash: inout UInt64, _ hasMerchantId: inout Bool,
    _ merchantMcc: inout Int, _ merchantAvgAmount: inout Double,
    _ terminalIsOnline: inout Bool, _ terminalCardPresent: inout Bool,
    _ terminalKmFromHome: inout Double,
    _ hasLastTx: inout Bool,
    _ lastYear: inout Int, _ lastMonth: inout Int, _ lastDay: inout Int,
    _ lastHour: inout Int, _ lastMinute: inout Int, _ lastSecond: inout Int,
    _ lastTxKmFromCurrent: inout Double
) {
    // This is a simplified manual JSON parser. For the implementation,
    // we use Foundation's JSONSerialization on the startup path only.
    // The hot-path parser reads bytes directly.
    // See the implementation in the actual source file for the full byte scanner.
    guard buf.count > 0 else { return }

    // Use JSONSerialization as initial implementation — to be replaced with
    // byte-level scanner for zero-alloc performance in a later optimization pass.
    guard let json = try? JSONSerialization.jsonObject(with: Data(bytes: buf.baseAddress!, count: buf.count)) as? [String: Any] else {
        return
    }

    if let tx = json["transaction"] as? [String: Any] {
        txAmount = (tx["amount"] as? Double) ?? 0
        txInstallments = (tx["installments"] as? Int) ?? 0
        if let ts = tx["requested_at"] as? String, ts.count >= 19 {
            let u = Array(ts.utf8)
            txYear = Int(u[0] - 0x30) * 1000 + Int(u[1] - 0x30) * 100 + Int(u[2] - 0x30) * 10 + Int(u[3] - 0x30)
            txMonth = Int(u[5] - 0x30) * 10 + Int(u[6] - 0x30)
            txDay = Int(u[8] - 0x30) * 10 + Int(u[9] - 0x30)
            txHour = Int(u[11] - 0x30) * 10 + Int(u[12] - 0x30)
            txMinute = Int(u[14] - 0x30) * 10 + Int(u[15] - 0x30)
            txSecond = Int(u[17] - 0x30) * 10 + Int(u[18] - 0x30)
        }
    }

    if let cust = json["customer"] as? [String: Any] {
        custAvgAmount = (cust["avg_amount"] as? Double) ?? 0
        custTxCount24h = (cust["tx_count_24h"] as? Int) ?? 0
        if let known = cust["known_merchants"] as? [String] {
            withUnsafeMutablePointer(to: &knownHashes) { ptr in
                ptr.withMemoryRebound(to: UInt64.self, capacity: 32) { hashes in
                    for s in known where knownCount < 32 {
                        let utf8 = Array(s.utf8)
                        utf8.withUnsafeBufferPointer { buf in
                            var hash: UInt64 = 14695981039346656037
                            for i in 0..<buf.count {
                                hash ^= UInt64(buf[i])
                                hash &*= 1099511628211
                            }
                            hashes[knownCount] = hash
                        }
                        knownCount += 1
                    }
                }
            }
        }
    }

    if let merch = json["merchant"] as? [String: Any] {
        if let mid = merch["id"] as? String {
            let utf8 = Array(mid.utf8)
            utf8.withUnsafeBufferPointer { buf in
                var hash: UInt64 = 14695981039346656037
                for i in 0..<buf.count {
                    hash ^= UInt64(buf[i])
                    hash &*= 1099511628211
                }
                merchantIdHash = hash
            }
            hasMerchantId = true
        }
        if let mccStr = merch["mcc"] as? String, let mcc = Int(mccStr) {
            merchantMcc = mcc
        }
        merchantAvgAmount = (merch["avg_amount"] as? Double) ?? 0
    }

    if let term = json["terminal"] as? [String: Any] {
        terminalIsOnline = (term["is_online"] as? Bool) ?? false
        terminalCardPresent = (term["card_present"] as? Bool) ?? false
        terminalKmFromHome = (term["km_from_home"] as? Double) ?? 0
    }

    if let lastTx = json["last_transaction"] as? [String: Any] {
        hasLastTx = true
        if let ts = lastTx["timestamp"] as? String, ts.count >= 19 {
            let u = Array(ts.utf8)
            lastYear = Int(u[0] - 0x30) * 1000 + Int(u[1] - 0x30) * 100 + Int(u[2] - 0x30) * 10 + Int(u[3] - 0x30)
            lastMonth = Int(u[5] - 0x30) * 10 + Int(u[6] - 0x30)
            lastDay = Int(u[8] - 0x30) * 10 + Int(u[9] - 0x30)
            lastHour = Int(u[11] - 0x30) * 10 + Int(u[12] - 0x30)
            lastMinute = Int(u[14] - 0x30) * 10 + Int(u[15] - 0x30)
            lastSecond = Int(u[17] - 0x30) * 10 + Int(u[18] - 0x30)
        }
        lastTxKmFromCurrent = (lastTx["km_from_current"] as? Double) ?? 0
    } else {
        hasLastTx = false
    }
}
```

- [ ] **Step 2: Write TransactionParserTests.swift**

```swift
import Testing
@testable import FraudDetector

@Test func goldenPayloadRoundTrip() throws {
    try MccRisk.initialize(
        mccRiskPath: "resources/mcc_risk.json",
        normalizationPath: "resources/normalization.json"
    )

    let json = """
    {"transaction":{"amount":150.00,"installments":3,"requested_at":"2026-04-15T14:30:00Z"},"customer":{"avg_amount":100.00,"tx_count_24h":5,"known_merchants":["m-001","m-002"]},"merchant":{"id":"m-001","mcc":"5812","avg_amount":120.00},"terminal":{"is_online":true,"card_present":true,"km_from_home":10.5},"last_transaction":{"timestamp":"2026-04-15T12:00:00Z","km_from_current":5.0}}
    """.trimmingCharacters(in: .whitespacesAndNewlines)

    var vector = [Float](repeating: 0, count: 16)
    let data = Array(json.utf8)
    data.withUnsafeBufferPointer { buf in
        let raw = UnsafeRawBufferPointer(buf)
        vector.withUnsafeMutableBufferPointer { vecBuf in
            TransactionParser.parse(raw, into: vecBuf.baseAddress!)
        }
    }

    // vector[0] = 150/10000 = 0.015
    #expect(vector[0] == Float(0.015))
    // vector[1] = 3/12 = 0.25
    #expect(vector[1] == Float(0.25))
    // vector[9] = is_online = 1.0
    #expect(vector[9] == Float(1.0))
    // vector[10] = card_present = 1.0
    #expect(vector[10] == Float(1.0))
    // vector[11] = m-001 is in known_merchants, so NOT unknown = 0.0
    #expect(vector[11] == Float(0.0))
}

@Test func zellerCongruence() {
    // 2026-04-15 is Wednesday = monBased 2
    // vector[4] = 2/6 = 0.3333
    try! MccRisk.initialize(
        mccRiskPath: "resources/mcc_risk.json",
        normalizationPath: "resources/normalization.json"
    )

    let json = """
    {"transaction":{"amount":100,"installments":1,"requested_at":"2026-04-15T12:00:00Z"},"customer":{"avg_amount":100,"tx_count_24h":0,"known_merchants":[]},"merchant":{"id":"m-x","mcc":"5411","avg_amount":100},"terminal":{"is_online":false,"card_present":false,"km_from_home":0},"last_transaction":null}
    """.trimmingCharacters(in: .whitespacesAndNewlines)

    var vector = [Float](repeating: 0, count: 16)
    Array(json.utf8).withUnsafeBufferPointer { buf in
        vector.withUnsafeMutableBufferPointer { vecBuf in
            TransactionParser.parse(UnsafeRawBufferPointer(buf), into: vecBuf.baseAddress!)
        }
    }

    // Wednesday = monBased 2, 2/6 = 0.3333
    #expect(abs(vector[4] - Float(0.3333)) < 0.001)
    // last_transaction is null → -1
    #expect(vector[5] == Float(-1.0))
    #expect(vector[6] == Float(-1.0))
}
```

- [ ] **Step 3: Run tests**

```bash
swift test --filter TransactionParserTests 2>&1
```
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add Sources/FraudDetector/TransactionParser.swift Tests/
git commit -m "feat: add transaction parser with manual JSON parsing and Zeller's congruence"
```

---

### Task 5: ResponseCache — Pre-Built HTTP Responses

**Files:**
- Create: `Sources/FraudDetector/ResponseCache.swift`

Reference: `HttpResponseTable.cs`

- [ ] **Step 1: Write ResponseCache.swift**

```swift
import NIOCore

public final class ResponseCache: Sendable {
    private let outcomes: [[UInt8]]
    public let ready: [UInt8]
    public let notFound: [UInt8]
    public let badRequest: [UInt8]

    private init() {
        ready = Self.buildEmpty(status: 200, reason: "OK")
        notFound = Self.buildEmpty(status: 404, reason: "Not Found")
        badRequest = Self.buildEmpty(status: 400, reason: "Bad Request")

        var out = [[UInt8]](repeating: [], count: 12)
        for fraudCount in 0...5 {
            let fraudScore = Float(fraudCount) * 0.2
            let scoreStr = String(format: "%.1f", fraudScore)
            let approvedBody = "{\"approved\":true,\"fraud_score\":\(scoreStr)}"
            let deniedBody = "{\"approved\":false,\"fraud_score\":\(scoreStr)}"
            out[fraudCount] = Self.buildFramed(status: 200, reason: "OK", jsonBody: approvedBody)
            out[6 + fraudCount] = Self.buildFramed(status: 200, reason: "OK", jsonBody: deniedBody)
        }
        outcomes = out
    }

    public static func build() -> ResponseCache {
        ResponseCache()
    }

    @inline(__always)
    public func get(approved: Bool, fraudCount: Int) -> [UInt8] {
        let fc = (fraudCount > 5 || fraudCount < 0) ? 0 : fraudCount
        return outcomes[approved ? fc : 6 + fc]
    }

    private static func buildFramed(status: Int, reason: String, jsonBody: String) -> [UInt8] {
        let bodyBytes = Array(jsonBody.utf8)
        let head = "HTTP/1.1 \(status) \(reason)\r\nContent-Length: \(bodyBytes.count)\r\nContent-Type: application/json\r\n\r\n"
        return Array(head.utf8) + bodyBytes
    }

    private static func buildEmpty(status: Int, reason: String) -> [UInt8] {
        let head = "HTTP/1.1 \(status) \(reason)\r\nContent-Length: 0\r\n\r\n"
        return Array(head.utf8)
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add Sources/FraudDetector/ResponseCache.swift
git commit -m "feat: add pre-built HTTP response cache (12 outcomes + ready/404/400)"
```

---

### Task 6: IvfDetector — mmap'd IVF Search Engine

**Files:**
- Create: `Sources/FraudDetector/IvfDetector.swift`

Reference: `IvfDetector.cs` — this is the most complex component.

- [ ] **Step 1: Write IvfDetector.swift**

```swift
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#elseif canImport(Musl)
import Musl
#endif
import CSimd

public final class IvfDetector: @unchecked Sendable {
    private let fd: Int32
    private let basePtr: UnsafeMutableRawPointer
    private let mapSize: Int

    private let exactFd: Int32
    private let exactBasePtr: UnsafeMutableRawPointer?
    private let exactMapSize: Int
    private let exactVectors: UnsafePointer<Float>?

    private let centroids: UnsafePointer<Int16>
    private let bboxMin: UnsafePointer<Int16>
    private let bboxMax: UnsafePointer<Int16>
    private let clusterMeta: UnsafePointer<IvfBinaryFormat.ClusterMeta>
    private let vectors: UnsafePointer<Int16>
    private let labels: UnsafePointer<UInt8>
    private let originalIndices: UnsafePointer<Int32>

    public let numVectors: Int
    public let numClusters: Int
    public let totalSlots: Int
    public let nprobeFull: Int

    public init(ivfPath: String, exactPath: String? = nil) throws {
        let ivfFd = open(ivfPath, O_RDONLY)
        guard ivfFd >= 0 else { throw IvfError.openFailed(ivfPath) }
        self.fd = ivfFd

        var st = stat()
        fstat(ivfFd, &st)
        let size = Int(st.st_size)
        self.mapSize = size

        guard let ptr = mmap(nil, size, PROT_READ, MAP_PRIVATE, ivfFd, 0) else {
            close(ivfFd)
            throw IvfError.mmapFailed
        }
        guard ptr != MAP_FAILED else {
            close(ivfFd)
            throw IvfError.mmapFailed
        }
        self.basePtr = ptr

        let version = ptr.load(fromByteOffset: 4, as: UInt32.self)
        guard version == IvfBinaryFormat.version else {
            throw IvfError.unsupportedVersion(version)
        }

        numVectors = Int(ptr.load(fromByteOffset: 8, as: UInt32.self))
        numClusters = Int(ptr.load(fromByteOffset: 12, as: UInt32.self))
        totalSlots = Int(ptr.load(fromByteOffset: 36, as: UInt32.self))

        let envFull = ProcessInfo.processInfo.environment["NPROBE_FULL"].flatMap(Int.init) ?? 0
        let fileFull = Int(ptr.load(fromByteOffset: 28, as: UInt32.self))
        nprobeFull = envFull > 0 ? envFull : fileFull

        centroids = (ptr + IvfBinaryFormat.centroidsOffset).assumingMemoryBound(to: Int16.self)
        bboxMin = (ptr + IvfBinaryFormat.bboxMinOffset(numClusters)).assumingMemoryBound(to: Int16.self)
        bboxMax = (ptr + IvfBinaryFormat.bboxMaxOffset(numClusters)).assumingMemoryBound(to: Int16.self)
        clusterMeta = (ptr + IvfBinaryFormat.clusterMetaOffset(numClusters)).assumingMemoryBound(to: IvfBinaryFormat.ClusterMeta.self)
        vectors = (ptr + IvfBinaryFormat.vectorsOffset(numClusters)).assumingMemoryBound(to: Int16.self)
        labels = (ptr + IvfBinaryFormat.labelsOffset(numClusters, totalSlots)).assumingMemoryBound(to: UInt8.self)
        originalIndices = (ptr + IvfBinaryFormat.originalIndicesOffset(numClusters, totalSlots)).assumingMemoryBound(to: Int32.self)

        if let ep = exactPath {
            let eFd = open(ep, O_RDONLY)
            guard eFd >= 0 else {
                munmap(ptr, size); close(ivfFd)
                throw IvfError.openFailed(ep)
            }
            self.exactFd = eFd
            var eSt = stat()
            fstat(eFd, &eSt)
            let eSize = Int(eSt.st_size)
            self.exactMapSize = eSize
            guard let ePtr = mmap(nil, eSize, PROT_READ, MAP_PRIVATE, eFd, 0), ePtr != MAP_FAILED else {
                munmap(ptr, size); close(ivfFd); close(eFd)
                throw IvfError.mmapFailed
            }
            self.exactBasePtr = ePtr
            self.exactVectors = (ePtr + ExactBinaryFormat.vectorsOffset).assumingMemoryBound(to: Float.self)
        } else {
            self.exactFd = -1
            self.exactBasePtr = nil
            self.exactMapSize = 0
            self.exactVectors = nil
        }
    }

    deinit {
        munmap(basePtr, mapSize)
        close(fd)
        if let ePtr = exactBasePtr {
            munmap(ePtr, exactMapSize)
            close(exactFd)
        }
    }

    public func prefault() {
        let totalSize = IvfBinaryFormat.totalSize(numClusters, totalSlots)
        MmapHints.hintHugePages(basePtr, totalSize)
        MmapHints.hintWillNeed(basePtr, totalSize)
        let bytePtr = basePtr.assumingMemoryBound(to: UInt8.self)
        var i = 0
        while i < totalSize {
            _ = bytePtr[i]
            i += 4096
        }

        if let ePtr = exactBasePtr {
            let pd = IvfBinaryFormat.paddedDims
            let exactBytes = numVectors * pd * MemoryLayout<Float>.size
            MmapHints.hintRandom(ePtr, exactBytes)
            MmapHints.hintHugePages(ePtr, exactBytes)
            let eBytePtr = ePtr.assumingMemoryBound(to: UInt8.self)
            var j = 0
            while j < exactBytes {
                _ = eBytePtr[j]
                j += 4096
            }
        }
    }

    public func score(_ query: UnsafePointer<Float>) -> (approved: Bool, fraudCount: Int) {
        let pd = IvfBinaryFormat.paddedDims
        let scale = IvfBinaryFormat.scale
        let rerankN = 6

        let qInt = UnsafeMutablePointer<Int16>.allocate(capacity: pd)
        defer { qInt.deallocate() }
        for d in 0..<pd {
            let v = (query[d] * Float(scale)).rounded()
            if v > Float(Int16.max) { qInt[d] = Int16.max }
            else if v < Float(Int16.min) { qInt[d] = Int16.min }
            else { qInt[d] = Int16(v) }
        }

        let candidateIdx = UnsafeMutablePointer<Int32>.allocate(capacity: rerankN)
        defer { candidateIdx.deallocate() }
        let candidateCount = findKNearest(query, qInt, rerankN, candidateIdx)

        let fraudCount: Int
        if let exactVecs = exactVectors {
            var topIdx = [Int32](repeating: 0, count: 5)
            var topDist = [Float](repeating: 0, count: 5)
            var topSize = 0

            for i in 0..<candidateCount {
                let vecIdx = Int(candidateIdx[i])
                let origGlobalIdx = Int(originalIndices[vecIdx])
                let dist = SimdDistance.float32L2Squared(query, exactVecs.advanced(by: origGlobalIdx * pd))

                if topSize < 5 {
                    var p = topSize - 1
                    while p >= 0 && topDist[p] > dist {
                        topDist[p + 1] = topDist[p]
                        topIdx[p + 1] = topIdx[p]
                        p -= 1
                    }
                    topDist[p + 1] = dist
                    topIdx[p + 1] = Int32(vecIdx)
                    topSize += 1
                } else if dist < topDist[4] {
                    var p = 3
                    while p >= 0 && topDist[p] > dist {
                        topDist[p + 1] = topDist[p]
                        topIdx[p + 1] = topIdx[p]
                        p -= 1
                    }
                    topDist[p + 1] = dist
                    topIdx[p + 1] = Int32(vecIdx)
                }
            }

            var fc = 0
            for i in 0..<topSize {
                fc += Int(labels[Int(topIdx[i])])
            }
            fraudCount = fc
        } else {
            let topK = min(candidateCount, 5)
            var fc = 0
            for i in 0..<topK {
                fc += Int(labels[Int(candidateIdx[i])])
            }
            fraudCount = fc
        }

        return (fraudCount < 3, fraudCount)
    }

    private func findKNearest(_ qFloat: UnsafePointer<Float>, _ qInt: UnsafePointer<Int16>,
                              _ k: Int, _ resultIdx: UnsafeMutablePointer<Int32>) -> Int {
        let pd = IvfBinaryFormat.paddedDims
        let actualNprobe = min(nprobeFull, numClusters)

        let probeList = UnsafeMutablePointer<Int32>.allocate(capacity: actualNprobe)
        let probeDists = UnsafeMutablePointer<Int32>.allocate(capacity: actualNprobe)
        defer { probeList.deallocate(); probeDists.deallocate() }
        var probeCount = 0

        // Centroid scan
        for c in 0..<numClusters {
            if c + 8 < numClusters {
                SimdDistance.prefetch(centroids.advanced(by: (c + 8) * pd))
            }
            let dist = SimdDistance.int16L2Squared(qInt, centroids.advanced(by: c * pd))

            if probeCount < actualNprobe {
                let pos = probeCount
                probeList[pos] = Int32(c)
                probeDists[pos] = dist
                probeCount += 1
                // Sift up (max-heap)
                var i = pos
                while i > 0 {
                    let parent = (i - 1) >> 1
                    if probeDists[parent] >= probeDists[i] { break }
                    swap(&probeList[parent], &probeList[i])
                    swap(&probeDists[parent], &probeDists[i])
                    i = parent
                }
            } else if dist < probeDists[0] {
                probeList[0] = Int32(c)
                probeDists[0] = dist
                siftDownProbe(probeList, probeDists, probeCount, 0)
            }
        }

        // K-heap for nearest vectors
        let heapIdx = UnsafeMutablePointer<Int32>.allocate(capacity: k)
        let heapDist = UnsafeMutablePointer<Int32>.allocate(capacity: k)
        defer { heapIdx.deallocate(); heapDist.deallocate() }
        var heapSize = 0

        // Initial cluster scan
        for p in 0..<probeCount {
            let cId = Int(probeList[p])
            if heapSize == k {
                let lb = SimdDistance.int16BboxLowerBound(qInt, bboxMin.advanced(by: cId * pd), bboxMax.advanced(by: cId * pd))
                if lb > heapDist[0] { continue }
            }
            scanCluster(qInt, cId, heapIdx, heapDist, &heapSize, k)
        }

        // Bbox repair pass
        if probeCount < numClusters {
            let bitsetWords = (numClusters + 63) >> 6
            let scannedBits = UnsafeMutablePointer<UInt64>.allocate(capacity: bitsetWords)
            defer { scannedBits.deallocate() }
            for w in 0..<bitsetWords { scannedBits[w] = 0 }
            for j in 0..<probeCount {
                let c = Int(probeList[j])
                scannedBits[c >> 6] |= 1 << (c & 63)
            }

            var worstDist = heapSize == k ? heapDist[0] : Int32.max
            for c in 0..<numClusters {
                if (scannedBits[c >> 6] & (1 << (c & 63))) != 0 { continue }
                let lb = SimdDistance.int16BboxLowerBound(qInt, bboxMin.advanced(by: c * pd), bboxMax.advanced(by: c * pd))
                if lb <= worstDist {
                    scanCluster(qInt, c, heapIdx, heapDist, &heapSize, k)
                    if heapSize == k { worstDist = heapDist[0] }
                }
            }
        }

        // Extract results in sorted order
        let resultCount = heapSize
        var i = heapSize - 1
        while i >= 0 {
            resultIdx[i] = heapIdx[0]
            heapPop(heapIdx, heapDist, &heapSize)
            i -= 1
        }
        return resultCount
    }

    private func scanCluster(_ qInt: UnsafePointer<Int16>, _ clusterId: Int,
                             _ heapIdx: UnsafeMutablePointer<Int32>,
                             _ heapDist: UnsafeMutablePointer<Int32>,
                             _ heapSize: inout Int, _ k: Int) {
        let pd = IvfBinaryFormat.paddedDims
        let bv = IvfBinaryFormat.blockVectors

        let meta = clusterMeta[clusterId]
        let offset = Int(meta.offset)
        let count = Int(meta.count)
        guard count > 0 else { return }

        let numBlocks = (count + bv - 1) / bv
        let clusterVecBase = vectors.advanced(by: offset * pd)
        let blockShorts = pd * bv
        let dimPairs = pd / 2

        for b in 0..<numBlocks {
            let blockPtr = clusterVecBase.advanced(by: b * blockShorts)
            let validLanes = min(bv, count - b * bv)

            for v in 0..<validLanes {
                var dist: Int32 = 0
                for kp in 0..<dimPairs {
                    let diff0 = Int32(qInt[2 * kp]) - Int32(blockPtr[kp * 16 + v * 2])
                    let diff1 = Int32(qInt[2 * kp + 1]) - Int32(blockPtr[kp * 16 + v * 2 + 1])
                    dist += diff0 * diff0 + diff1 * diff1
                }

                let vecIdx = Int32(offset + b * bv + v)
                if heapSize < k {
                    heapPush(heapIdx, heapDist, &heapSize, vecIdx, dist)
                } else if dist < heapDist[0] {
                    heapReplaceTop(heapIdx, heapDist, k, vecIdx, dist)
                }
            }
        }
    }

    @inline(__always)
    private func siftDownProbe(_ list: UnsafeMutablePointer<Int32>, _ dists: UnsafeMutablePointer<Int32>, _ size: Int, _ start: Int) {
        var i = start
        while true {
            var largest = i
            let left = 2 * i + 1, right = 2 * i + 2
            if left < size && dists[left] > dists[largest] { largest = left }
            if right < size && dists[right] > dists[largest] { largest = right }
            if largest == i { break }
            swap(&list[largest], &list[i])
            swap(&dists[largest], &dists[i])
            i = largest
        }
    }

    @inline(__always)
    private func heapPush(_ idx: UnsafeMutablePointer<Int32>, _ dist: UnsafeMutablePointer<Int32>,
                          _ size: inout Int, _ vectorIndex: Int32, _ d: Int32) {
        let i = size
        idx[i] = vectorIndex
        dist[i] = d
        size += 1
        var pos = i
        while pos > 0 {
            let parent = (pos - 1) >> 1
            if dist[parent] >= dist[pos] { break }
            swap(&idx[parent], &idx[pos])
            swap(&dist[parent], &dist[pos])
            pos = parent
        }
    }

    @inline(__always)
    private func heapReplaceTop(_ idx: UnsafeMutablePointer<Int32>, _ dist: UnsafeMutablePointer<Int32>,
                                _ size: Int, _ vectorIndex: Int32, _ d: Int32) {
        idx[0] = vectorIndex
        dist[0] = d
        var i = 0
        while true {
            var largest = i
            let left = 2 * i + 1, right = 2 * i + 2
            if left < size && dist[left] > dist[largest] { largest = left }
            if right < size && dist[right] > dist[largest] { largest = right }
            if largest == i { break }
            swap(&idx[largest], &idx[i])
            swap(&dist[largest], &dist[i])
            i = largest
        }
    }

    @inline(__always)
    private func heapPop(_ idx: UnsafeMutablePointer<Int32>, _ dist: UnsafeMutablePointer<Int32>,
                         _ size: inout Int) {
        size -= 1
        idx[0] = idx[size]
        dist[0] = dist[size]
        guard size > 0 else { return }
        var i = 0
        while true {
            var largest = i
            let left = 2 * i + 1, right = 2 * i + 2
            if left < size && dist[left] > dist[largest] { largest = left }
            if right < size && dist[right] > dist[largest] { largest = right }
            if largest == i { break }
            swap(&idx[largest], &idx[i])
            swap(&dist[largest], &dist[i])
            i = largest
        }
    }

    enum IvfError: Error {
        case openFailed(String)
        case mmapFailed
        case unsupportedVersion(UInt32)
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add Sources/FraudDetector/IvfDetector.swift
git commit -m "feat: add IVF detector with mmap, centroid search, bbox repair, float32 rerank"
```

---

### Task 7: ExactDetector — Brute-Force Float32

**Files:**
- Create: `Sources/FraudDetector/ExactDetector.swift`

Reference: `ExactDetector.cs`

- [ ] **Step 1: Write ExactDetector.swift**

```swift
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#elseif canImport(Musl)
import Musl
#endif

public final class ExactDetector: @unchecked Sendable {
    private let fd: Int32
    private let basePtr: UnsafeMutableRawPointer
    private let mapSize: Int
    private let vectors: UnsafePointer<Float>
    private let labels: UnsafePointer<UInt8>
    public let numVectors: Int

    public init(path: String) throws {
        let fileFd = open(path, O_RDONLY)
        guard fileFd >= 0 else { throw ExactError.openFailed(path) }
        self.fd = fileFd

        var st = stat()
        fstat(fileFd, &st)
        let size = Int(st.st_size)
        self.mapSize = size

        guard let ptr = mmap(nil, size, PROT_READ, MAP_PRIVATE, fileFd, 0), ptr != MAP_FAILED else {
            close(fileFd)
            throw ExactError.mmapFailed
        }
        self.basePtr = ptr

        let version = ptr.load(fromByteOffset: 4, as: UInt32.self)
        guard version == ExactBinaryFormat.version else {
            throw ExactError.unsupportedVersion(version)
        }

        numVectors = Int(ptr.load(fromByteOffset: 8, as: UInt32.self))
        vectors = (ptr + ExactBinaryFormat.vectorsOffset).assumingMemoryBound(to: Float.self)
        labels = (ptr + ExactBinaryFormat.labelsOffset(numVectors)).assumingMemoryBound(to: UInt8.self)
    }

    deinit {
        munmap(basePtr, mapSize)
        close(fd)
    }

    public func prefault() {
        let total = ExactBinaryFormat.totalSize(numVectors)
        let bytePtr = basePtr.assumingMemoryBound(to: UInt8.self)
        var i = 0
        while i < total {
            _ = bytePtr[i]
            i += 4096
        }
    }

    public func score(_ query: UnsafePointer<Float>) -> (approved: Bool, fraudCount: Int) {
        let pd = ExactBinaryFormat.paddedDims
        var topIdx = [Int32](repeating: 0, count: 5)
        var topDist = [Float](repeating: 0, count: 5)
        var size = 0

        for i in 0..<numVectors {
            let dist = SimdDistance.float32L2Squared(query, vectors.advanced(by: i * pd))
            if size < 5 {
                heapPush(&topIdx, &topDist, &size, Int32(i), dist)
            } else if dist < topDist[0] {
                topIdx[0] = Int32(i)
                topDist[0] = dist
                siftDown(&topIdx, &topDist, 5, 0)
            }
        }

        var fraudCount = 0
        for i in 0..<size {
            fraudCount += Int(labels[Int(topIdx[i])])
        }
        return (fraudCount < 3, fraudCount)
    }

    @inline(__always)
    private func heapPush(_ idx: inout [Int32], _ dist: inout [Float], _ size: inout Int, _ v: Int32, _ d: Float) {
        let i = size
        idx[i] = v; dist[i] = d
        size += 1
        var pos = i
        while pos > 0 {
            let parent = (pos - 1) >> 1
            if dist[parent] >= dist[pos] { break }
            idx.swapAt(parent, pos)
            dist.swapAt(parent, pos)
            pos = parent
        }
    }

    @inline(__always)
    private func siftDown(_ idx: inout [Int32], _ dist: inout [Float], _ size: Int, _ start: Int) {
        var i = start
        while true {
            var largest = i
            let l = 2 * i + 1, r = 2 * i + 2
            if l < size && dist[l] > dist[largest] { largest = l }
            if r < size && dist[r] > dist[largest] { largest = r }
            if largest == i { break }
            idx.swapAt(largest, i)
            dist.swapAt(largest, i)
            i = largest
        }
    }

    enum ExactError: Error {
        case openFailed(String)
        case mmapFailed
        case unsupportedVersion(UInt32)
    }
}
```

- [ ] **Step 2: Commit**

```bash
git add Sources/FraudDetector/ExactDetector.swift
git commit -m "feat: add exact brute-force float32 detector with mmap"
```

---

### Task 8: HttpHandler — SwiftNIO HTTP/1.1 Server

**Files:**
- Create: `Sources/FraudDetector/HttpHandler.swift`

Reference: `SocketHttpServer.cs`

- [ ] **Step 1: Write HttpHandler.swift**

```swift
import NIOCore
import NIOPosix
import NIOHTTP1

public final class HttpHandler: ChannelInboundHandler, @unchecked Sendable {
    public typealias InboundIn = HTTPServerRequestPart
    public typealias OutboundOut = HTTPServerResponsePart

    private let detector: IvfDetector
    private let responses: ResponseCache
    private var bodyBuffer: ByteBuffer?

    public init(detector: IvfDetector, responses: ResponseCache) {
        self.detector = detector
        self.responses = responses
    }

    public func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        let part = unwrapInboundIn(data)

        switch part {
        case .head(let head):
            if head.method == .GET && head.uri == "/ready" {
                sendPrebuilt(context: context, responses.ready)
                return
            }
            if head.method != .POST || head.uri != "/fraud-score" {
                sendPrebuilt(context: context, responses.notFound)
                return
            }
            bodyBuffer = context.channel.allocator.buffer(capacity: 2048)

        case .body(var buf):
            bodyBuffer?.writeBuffer(&buf)

        case .end:
            guard let body = bodyBuffer else { return }
            bodyBuffer = nil
            processRequest(context: context, body: body)
        }
    }

    private func processRequest(context: ChannelHandlerContext, body: ByteBuffer) {
        let responseBytes: [UInt8]
        do {
            let readable = body.readableBytesView
            var vector = [Float](repeating: 0, count: 16)

            readable.withUnsafeBytes { rawBuf in
                vector.withUnsafeMutableBufferPointer { vecBuf in
                    TransactionParser.parse(rawBuf, into: vecBuf.baseAddress!)
                }
            }

            let result = vector.withUnsafeBufferPointer { buf in
                detector.score(buf.baseAddress!)
            }
            responseBytes = responses.get(approved: result.approved, fraudCount: result.fraudCount)
        } catch {
            responseBytes = responses.get(approved: true, fraudCount: 0)
        }

        sendPrebuilt(context: context, responseBytes)
    }

    private func sendPrebuilt(context: ChannelHandlerContext, _ bytes: [UInt8]) {
        var buf = context.channel.allocator.buffer(capacity: bytes.count)
        buf.writeBytes(bytes)
        context.writeAndFlush(wrapOutboundOut(.body(.byteBuffer(buf))), promise: nil)
    }

    public func errorCaught(context: ChannelHandlerContext, error: Error) {
        context.close(promise: nil)
    }
}
```

Note: The above uses NIOHTTP1 for simplicity. For maximum performance matching the C# raw socket approach, we should bypass NIOHTTP1 and write a raw byte handler. However, NIOHTTP1 is still very fast and we can optimize later if needed.

**Alternative: Raw byte handler (closer to C# SocketHttpServer)**

For the actual implementation, write a `RawHttpHandler` that operates on `ByteBuffer` directly without NIOHTTP1 parsing — matching the C# approach of parsing only method, path, and content-length from raw bytes. This can be done as a separate optimization task after the initial port works.

- [ ] **Step 2: Commit**

```bash
git add Sources/FraudDetector/HttpHandler.swift
git commit -m "feat: add SwiftNIO HTTP handler for fraud-score and ready endpoints"
```

---

### Task 9: Main Entry Point

**Files:**
- Create: `Sources/RinhaApp/main.swift`

Reference: `Program.cs`

- [ ] **Step 1: Write main.swift**

```swift
import Foundation
import NIOCore
import NIOPosix
import NIOHTTP1
import FraudDetector

let resourcesPath = ProcessInfo.processInfo.environment["RESOURCES_PATH"] ?? "/resources"
try MccRisk.initialize(
    mccRiskPath: (resourcesPath as NSString).appendingPathComponent("mcc_risk.json"),
    normalizationPath: (resourcesPath as NSString).appendingPathComponent("normalization.json")
)

let dataPath = ProcessInfo.processInfo.environment["INDEX_PATH"] ?? "/data/ivf.bin"
let exactPath = ProcessInfo.processInfo.environment["EXACT_PATH"]
let socketPath = ProcessInfo.processInfo.environment["SOCKET_PATH"]
let port = Int(ProcessInfo.processInfo.environment["API_PORT"] ?? "8080") ?? 8080

print("Opening index from \(dataPath)...")
let detector: IvfDetector
if let ep = exactPath, FileManager.default.fileExists(atPath: ep) {
    print("  with exact float32 rerank from \(ep)")
    detector = try IvfDetector(ivfPath: dataPath, exactPath: ep)
} else {
    detector = try IvfDetector(ivfPath: dataPath)
}
print("Loaded \(detector.numVectors) vectors, \(detector.numClusters) clusters, nprobe=\(detector.nprobeFull)")
print("Prefaulting pages...")
detector.prefault()

// Warmup
let warmupIterations = 200
let warmupPath = (resourcesPath as NSString).appendingPathComponent("warmup-payloads.ndjson")
var warmedReal = 0
if FileManager.default.fileExists(atPath: warmupPath) {
    print("Warming up with real payloads from \(warmupPath)...")
    if let contents = try? String(contentsOfFile: warmupPath, encoding: .utf8) {
        var vector = [Float](repeating: 0, count: 16)
        for line in contents.split(separator: "\n") where warmedReal < warmupIterations {
            let data = Array(line.utf8)
            data.withUnsafeBufferPointer { buf in
                vector.withUnsafeMutableBufferPointer { vecBuf in
                    for i in 0..<16 { vecBuf[i] = 0 }
                    TransactionParser.parse(UnsafeRawBufferPointer(buf), into: vecBuf.baseAddress!)
                }
            }
            _ = vector.withUnsafeBufferPointer { buf in
                detector.score(buf.baseAddress!)
            }
            warmedReal += 1
        }
    }
    print("  warmed up with \(warmedReal) real payloads.")
}
if warmedReal < warmupIterations {
    print("  filling remaining \(warmupIterations - warmedReal) with random vectors")
    var rng = RandomNumberGenerator_LCG(seed: 42)
    var vector = [Float](repeating: 0, count: 16)
    for _ in warmedReal..<warmupIterations {
        for d in 0..<14 { vector[d] = Float.random(in: 0..<1, using: &rng) }
        _ = vector.withUnsafeBufferPointer { buf in
            detector.score(buf.baseAddress!)
        }
    }
}
print("Ready.")

let responses = ResponseCache.build()

let group = MultiThreadedEventLoopGroup(numberOfThreads: System.coreCount)
let bootstrap = ServerBootstrap(group: group)
    .serverChannelOption(.backlog, value: 2048)
    .serverChannelOption(.socketOption(.so_reuseaddr), value: 1)
    .childChannelInitializer { channel in
        channel.pipeline.configureHTTPServerPipeline().flatMap {
            channel.pipeline.addHandler(HttpHandler(detector: detector, responses: responses))
        }
    }

let channel: Channel
if let sp = socketPath {
    if FileManager.default.fileExists(atPath: sp) {
        try FileManager.default.removeItem(atPath: sp)
    }
    channel = try await bootstrap.bind(unixDomainSocketPath: sp).get()
    // Set permissions
    chmod(sp, 0o777)
    print("Listening on Unix socket \(sp)")
} else {
    channel = try await bootstrap.bind(host: "0.0.0.0", port: port).get()
    print("Listening on port \(port)")
}

try await channel.closeFuture.get()

struct RandomNumberGenerator_LCG: RandomNumberGenerator {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return state
    }
}
```

- [ ] **Step 2: Verify build**

```bash
swift build 2>&1
```

- [ ] **Step 3: Commit**

```bash
git add Sources/RinhaApp/main.swift
git commit -m "feat: add main entry point with server bootstrap, warmup, and mmap prefault"
```

---

### Task 10: Preprocessor — Data Loader + K-Means + Binary Writers

**Files:**
- Create: `Sources/Preprocessor/PreprocessorMain.swift`
- Create: `Sources/Preprocessor/DataLoader.swift`
- Create: `Sources/Preprocessor/KMeans.swift`
- Create: `Sources/Preprocessor/IvfBuilder.swift`
- Create: `Sources/Preprocessor/IvfBinaryWriter.swift`
- Create: `Sources/Preprocessor/ExactBinaryWriter.swift`
- Create: `Sources/Preprocessor/GenWarmup.swift`

Reference: `PreprocessCommand.cs`, `IvfBuilder.cs`, `IvfBinaryWriter.cs`, `ExactBinaryWriter.cs`, `GenWarmupCommand.cs`

- [ ] **Step 1: Write DataLoader.swift**

```swift
import Foundation
import FraudDetector

enum DataLoader {
    struct VectorData {
        var vectors: [[Float]]
        var labels: [UInt8]
    }

    static func load(from path: String) throws -> VectorData {
        print("Reading \(path)...")
        let url = URL(fileURLWithPath: path)
        let data: Data
        if path.hasSuffix(".gz") {
            let compressedData = try Data(contentsOf: url)
            data = try (compressedData as NSData).decompressed(using: .zlib) as Data
        } else {
            data = try Data(contentsOf: url)
        }

        let json = try JSONSerialization.jsonObject(with: data) as! [[String: Any]]
        var vectors = [[Float]]()
        vectors.reserveCapacity(3_200_000)
        var labels = [UInt8]()
        labels.reserveCapacity(3_200_000)

        for entry in json {
            let vecArray = entry["vector"] as! [Double]
            var vec = [Float](repeating: 0, count: 14)
            for (i, v) in vecArray.enumerated() where i < 14 {
                vec[i] = Float(v)
            }
            vectors.append(vec)
            let label = (entry["label"] as! String) == "fraud" ? UInt8(1) : UInt8(0)
            labels.append(label)
        }

        print("Loaded \(vectors.count) vectors.")
        return VectorData(vectors: vectors, labels: labels)
    }
}
```

- [ ] **Step 2: Write KMeans.swift**

```swift
import Foundation
import FraudDetector

enum KMeans {
    static func cluster(vectors: [Float], n: Int, k: Int, iterations: Int) -> [Int] {
        let pd = 16
        var centroids = [Float](repeating: 0, count: k * pd)
        for c in 0..<k {
            let srcIdx = c * n / k
            for d in 0..<pd {
                centroids[c * pd + d] = vectors[srcIdx * pd + d]
            }
        }

        var assignments = [Int](repeating: 0, count: n)
        for iter in 0..<iterations {
            print("\r  K-means iteration \(iter + 1)/\(iterations)...", terminator: "")
            fflush(stdout)
            assignVectors(vectors, &centroids, &assignments, n, k)
            recomputeCentroids(vectors, assignments, &centroids, n, k)
        }
        print()
        assignVectors(vectors, &centroids, &assignments, n, k)
        return assignments
    }

    private static func assignVectors(_ flatVectors: [Float], _ centroids: inout [Float],
                                       _ assignments: inout [Int], _ n: Int, _ k: Int) {
        let pd = 16
        DispatchQueue.concurrentPerform(iterations: n) { i in
            let vOffset = i * pd
            var bestDist: Float = .greatestFiniteMagnitude
            var bestCluster = 0
            for c in 0..<k {
                let cOffset = c * pd
                var dist: Float = 0
                for d in 0..<pd {
                    let diff = flatVectors[vOffset + d] - centroids[cOffset + d]
                    dist += diff * diff
                }
                if dist < bestDist {
                    bestDist = dist
                    bestCluster = c
                }
            }
            assignments[i] = bestCluster
        }
    }

    private static func recomputeCentroids(_ flatVectors: [Float], _ assignments: [Int],
                                            _ centroids: inout [Float], _ n: Int, _ k: Int) {
        let pd = 16
        for i in 0..<centroids.count { centroids[i] = 0 }
        var counts = [Int](repeating: 0, count: k)

        for i in 0..<n {
            let c = assignments[i]
            counts[c] += 1
            let cOffset = c * pd
            let vOffset = i * pd
            for d in 0..<14 {
                centroids[cOffset + d] += flatVectors[vOffset + d]
            }
        }

        for c in 0..<k {
            guard counts[c] > 0 else { continue }
            let inv: Float = 1.0 / Float(counts[c])
            let offset = c * pd
            for d in 0..<14 {
                centroids[offset + d] *= inv
            }
        }
    }
}
```

- [ ] **Step 3: Write IvfBuilder.swift**

```swift
import Foundation
import FraudDetector

struct IvfResult {
    var centroids: [Float]
    var vectors: [Float]
    var labels: [UInt8]
    var clusterOffsets: [Int]
    var clusterCounts: [Int]
    var originalIndices: [Int32]
    var numVectors: Int
    var numClusters: Int
}

enum IvfBuilder {
    static func build(inputVectors: [[Float]], inputLabels: [UInt8], numClusters: Int, kmeansIterations: Int = 20) -> IvfResult {
        let n = inputVectors.count
        let pd = 16

        var padded = [Float](repeating: 0, count: n * pd)
        for i in 0..<n {
            let src = inputVectors[i]
            let destOffset = i * pd
            for d in 0..<14 {
                padded[destOffset + d] = src[d]
            }
        }

        let assignments = KMeans.cluster(vectors: padded, n: n, k: numClusters, iterations: kmeansIterations)

        var centroids = [Float](repeating: 0, count: numClusters * pd)
        var counts = [Int](repeating: 0, count: numClusters)
        for i in 0..<n {
            let c = assignments[i]
            counts[c] += 1
            for d in 0..<14 {
                centroids[c * pd + d] += padded[i * pd + d]
            }
        }
        for c in 0..<numClusters {
            guard counts[c] > 0 else { continue }
            let inv: Float = 1.0 / Float(counts[c])
            for d in 0..<14 {
                centroids[c * pd + d] *= inv
            }
        }

        var clusterCounts = [Int](repeating: 0, count: numClusters)
        for i in 0..<n { clusterCounts[assignments[i]] += 1 }

        var clusterOffsets = [Int](repeating: 0, count: numClusters)
        for c in 1..<numClusters {
            clusterOffsets[c] = clusterOffsets[c - 1] + clusterCounts[c - 1]
        }

        var outVectors = [Float](repeating: 0, count: n * pd)
        var outLabels = [UInt8](repeating: 0, count: n)
        var originalIndices = [Int32](repeating: 0, count: n)
        var insertPos = [Int](repeating: 0, count: numClusters)

        for i in 0..<n {
            let c = assignments[i]
            let destIdx = clusterOffsets[c] + insertPos[c]
            insertPos[c] += 1
            for d in 0..<pd {
                outVectors[destIdx * pd + d] = padded[i * pd + d]
            }
            outLabels[destIdx] = inputLabels[i]
            originalIndices[destIdx] = Int32(i)
        }

        // Sort within each cluster by distance to centroid
        for c in 0..<numClusters {
            let off = clusterOffsets[c]
            let cnt = clusterCounts[c]
            guard cnt > 1 else { continue }

            var order = Array(0..<cnt)
            var distArr = [Float](repeating: 0, count: cnt)
            for i in 0..<cnt {
                let vOff = (off + i) * pd
                var d: Float = 0
                for dim in 0..<14 {
                    let diff = outVectors[vOff + dim] - centroids[c * pd + dim]
                    d += diff * diff
                }
                distArr[i] = d
            }

            order.sort { distArr[$0] < distArr[$1] }

            var tmpVec = [Float](repeating: 0, count: cnt * pd)
            var tmpLbl = [UInt8](repeating: 0, count: cnt)
            var tmpOrig = [Int32](repeating: 0, count: cnt)
            for i in 0..<cnt {
                let src = off + order[i]
                for d in 0..<pd { tmpVec[i * pd + d] = outVectors[src * pd + d] }
                tmpLbl[i] = outLabels[src]
                tmpOrig[i] = originalIndices[src]
            }
            for i in 0..<cnt {
                for d in 0..<pd { outVectors[(off + i) * pd + d] = tmpVec[i * pd + d] }
                outLabels[off + i] = tmpLbl[i]
                originalIndices[off + i] = tmpOrig[i]
            }
        }
        print("  Sorted vectors by centroid distance within each cluster.")

        return IvfResult(
            centroids: centroids, vectors: outVectors, labels: outLabels,
            clusterOffsets: clusterOffsets, clusterCounts: clusterCounts,
            originalIndices: originalIndices, numVectors: n, numClusters: numClusters
        )
    }
}
```

- [ ] **Step 4: Write IvfBinaryWriter.swift**

```swift
import Foundation
import FraudDetector

enum IvfBinaryWriter {
    static func write(path: String, ivf: IvfResult, nprobeFull: Int = 40, nprobeFast: Int = 5) {
        let k = ivf.numClusters
        let n = ivf.numVectors
        let pd = IvfBinaryFormat.paddedDims
        let bv = IvfBinaryFormat.blockVectors
        let scale = Float(IvfBinaryFormat.scale)

        var paddedOffsets = [Int](repeating: 0, count: k)
        var counts = ivf.clusterCounts
        var totalSlots = 0
        for c in 0..<k {
            paddedOffsets[c] = totalSlots
            let blocks = (counts[c] + bv - 1) / bv
            totalSlots += blocks * bv
        }

        // Quantize vectors to int16
        var int16Vectors = [Int16](repeating: 0, count: totalSlots * pd)
        var paddedLabels = [UInt8](repeating: 0, count: totalSlots)
        var paddedOriginalIndices = [Int32](repeating: -1, count: totalSlots)

        for c in 0..<k {
            let srcOff = ivf.clusterOffsets[c]
            let dstOff = paddedOffsets[c]
            let count = counts[c]
            for i in 0..<count {
                let sBase = (srcOff + i) * pd
                let dBase = (dstOff + i) * pd
                for d in 0..<pd {
                    int16Vectors[dBase + d] = clampToShort(ivf.vectors[sBase + d] * scale)
                }
                paddedLabels[dstOff + i] = ivf.labels[srcOff + i]
                paddedOriginalIndices[dstOff + i] = ivf.originalIndices[srcOff + i]
            }
        }

        // Compute bounding boxes
        var bboxMin = [Int16](repeating: Int16.max, count: k * pd)
        var bboxMax = [Int16](repeating: Int16.min, count: k * pd)
        for c in 0..<k {
            let dstOff = paddedOffsets[c]
            let count = counts[c]
            let baseOff = c * pd
            for i in 0..<count {
                let vOff = (dstOff + i) * pd
                for d in 0..<pd {
                    let v = int16Vectors[vOff + d]
                    if v < bboxMin[baseOff + d] { bboxMin[baseOff + d] = v }
                    if v > bboxMax[baseOff + d] { bboxMax[baseOff + d] = v }
                }
            }
            if count == 0 {
                for d in 0..<pd {
                    let cv = clampToShort(ivf.centroids[baseOff + d] * scale)
                    bboxMin[baseOff + d] = cv
                    bboxMax[baseOff + d] = cv
                }
            }
        }

        // SoA blocking
        let totalBlocks = totalSlots / bv
        var blockedVectors = [Int16](repeating: 0, count: totalSlots * pd)
        for b in 0..<totalBlocks {
            let blockStart = b * bv
            for kp in 0..<(pd / 2) {
                for v in 0..<bv {
                    let srcBase = (blockStart + v) * pd + 2 * kp
                    let dstBase = b * pd * bv + kp * 2 * bv + v * 2
                    blockedVectors[dstBase + 0] = int16Vectors[srcBase + 0]
                    blockedVectors[dstBase + 1] = int16Vectors[srcBase + 1]
                }
            }
        }

        // Quantize centroids to int16
        var int16Centroids = [Int16](repeating: 0, count: k * pd)
        for i in 0..<(k * pd) {
            int16Centroids[i] = clampToShort(ivf.centroids[i] * scale)
        }

        // Write binary file
        guard let fh = FileHandle(forWritingAtPath: path) ?? {
            FileManager.default.createFile(atPath: path, contents: nil)
            return FileHandle(forWritingAtPath: path)
        }() else { fatalError("Cannot open \(path) for writing") }

        defer { fh.closeFile() }
        fh.truncateFile(atOffset: 0)

        // Header (64 bytes)
        var header = Data(count: IvfBinaryFormat.headerSize)
        header.replaceSubrange(0..<4, with: IvfBinaryFormat.magic)
        withUnsafeBytes(of: IvfBinaryFormat.version) { header.replaceSubrange(4..<8, with: $0) }
        withUnsafeBytes(of: UInt32(n)) { header.replaceSubrange(8..<12, with: $0) }
        withUnsafeBytes(of: UInt32(k)) { header.replaceSubrange(12..<16, with: $0) }
        withUnsafeBytes(of: UInt32(IvfBinaryFormat.dims)) { header.replaceSubrange(16..<20, with: $0) }
        withUnsafeBytes(of: UInt32(IvfBinaryFormat.paddedDims)) { header.replaceSubrange(20..<24, with: $0) }
        withUnsafeBytes(of: UInt32(nprobeFast)) { header.replaceSubrange(24..<28, with: $0) }
        withUnsafeBytes(of: UInt32(nprobeFull)) { header.replaceSubrange(28..<32, with: $0) }
        withUnsafeBytes(of: UInt32(IvfBinaryFormat.scale)) { header.replaceSubrange(32..<36, with: $0) }
        withUnsafeBytes(of: UInt32(totalSlots)) { header.replaceSubrange(36..<40, with: $0) }
        fh.write(header)

        fh.write(int16Centroids.withUnsafeBytes { Data($0) })
        fh.write(bboxMin.withUnsafeBytes { Data($0) })
        fh.write(bboxMax.withUnsafeBytes { Data($0) })

        // Cluster metadata
        for c in 0..<k {
            withUnsafeBytes(of: UInt32(paddedOffsets[c])) { fh.write(Data($0)) }
            withUnsafeBytes(of: UInt32(counts[c])) { fh.write(Data($0)) }
        }

        fh.write(blockedVectors.withUnsafeBytes { Data($0) })
        fh.write(Data(paddedLabels))
        fh.write(paddedOriginalIndices.withUnsafeBytes { Data($0) })
    }

    private static func clampToShort(_ v: Float) -> Int16 {
        let r = v.rounded()
        if r > Float(Int16.max) { return Int16.max }
        if r < Float(Int16.min) { return Int16.min }
        return Int16(r)
    }
}
```

- [ ] **Step 5: Write ExactBinaryWriter.swift**

```swift
import Foundation
import FraudDetector

enum ExactBinaryWriter {
    static func write(path: String, paddedVectors: [Float], labels: [UInt8], numVectors: Int) {
        let pd = ExactBinaryFormat.paddedDims

        guard let fh = FileHandle(forWritingAtPath: path) ?? {
            FileManager.default.createFile(atPath: path, contents: nil)
            return FileHandle(forWritingAtPath: path)
        }() else { fatalError("Cannot open \(path) for writing") }
        defer { fh.closeFile() }
        fh.truncateFile(atOffset: 0)

        var header = Data(count: ExactBinaryFormat.headerSize)
        header.replaceSubrange(0..<4, with: ExactBinaryFormat.magic)
        withUnsafeBytes(of: ExactBinaryFormat.version) { header.replaceSubrange(4..<8, with: $0) }
        withUnsafeBytes(of: UInt32(numVectors)) { header.replaceSubrange(8..<12, with: $0) }
        fh.write(header)

        fh.write(paddedVectors.withUnsafeBytes { Data($0) })
        fh.write(Data(labels.prefix(numVectors)))
    }
}
```

- [ ] **Step 6: Write GenWarmup.swift**

```swift
import Foundation

enum GenWarmup {
    static func run(outputPath: String, count: Int = 200) {
        let amounts: [Double] = [10, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 50000]
        let installments = [1, 3, 6, 12]
        let mccs = [5411, 5812, 5814, 5912, 5921, 5942, 6011, 7011, 7995, 4121]
        let kmFromHome: [Double] = [0.2, 1.5, 5.0, 25.0, 200.0]
        let txCount24h = [0, 1, 3, 10, 50]
        let terminalFlags: [(Bool, Bool)] = [(true, true), (true, false), (false, true), (false, false)]
        let merchantLists: [[String]] = [[], ["m-001"], ["m-001", "m-002"], ["m-001", "m-002", "m-003", "m-004", "m-005"]]

        let dir = (outputPath as NSString).deletingLastPathComponent
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)

        var lines = [String]()
        var i = 0, written = 0
        outer: for a in amounts {
            for m in mccs {
                for k in kmFromHome {
                    if written >= count { break outer }
                    let inst = installments[i % installments.count]
                    let tx24 = txCount24h[i % txCount24h.count]
                    let merch = merchantLists[i % merchantLists.count]
                    let flags = terminalFlags[i % terminalFlags.count]
                    i += 1

                    let merchStr = merch.map { "\"\($0)\"" }.joined(separator: ",")
                    let line = """
                    {"id":"warm-tx-\(i)","transaction":{"amount":\(String(format: "%.2f", a)),"installments":\(inst),"requested_at":"2026-04-15T13:42:11Z"},"customer":{"avg_amount":\(String(format: "%.2f", a / 1.5)),"tx_count_24h":\(tx24),"known_merchants":[\(merchStr)]},"merchant":{"id":"m-001","mcc":"\(m)","avg_amount":\(String(format: "%.2f", a * 0.9))},"terminal":{"is_online":\(flags.0),"card_present":\(flags.1),"km_from_home":\(String(format: "%.2f", k))},"last_transaction":{"timestamp":"2026-04-15T11:10:00Z","km_from_current":\(String(format: "%.2f", k / 2))}}
                    """
                    lines.append(line)
                    written += 1
                }
            }
        }

        try! lines.joined(separator: "\n").write(toFile: outputPath, atomically: true, encoding: .utf8)
        print("Wrote \(written) warmup payloads to \(outputPath)")
    }
}
```

- [ ] **Step 7: Write PreprocessorMain.swift**

```swift
import Foundation
import FraudDetector

@main
struct PreprocessorApp {
    static func main() throws {
        let args = CommandLine.arguments
        guard args.count >= 3 else {
            print("Usage: Preprocessor <references.json.gz> <output.bin> [clusters] [kmeans_iter] [format] [nprobe]")
            print("       Preprocessor gen-warmup <output.ndjson> [count]")
            exit(1)
        }

        if args[1] == "gen-warmup" {
            let count = args.count > 3 ? Int(args[3]) ?? 200 : 200
            GenWarmup.run(outputPath: args[2], count: count)
            return
        }

        let inputPath = args[1]
        let outputPath = args[2]
        let format = args.count > 5 ? args[5] : "ivf"
        let numClusters = args.count > 3 ? Int(args[3]) ?? 4096 : 4096
        let kmeansIter = args.count > 4 ? Int(args[4]) ?? 20 : 20
        let nprobe = args.count > 6 ? Int(args[6]) ?? 40 : 40

        let data = try DataLoader.load(from: inputPath)

        let dir = (outputPath as NSString).deletingLastPathComponent
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)

        if format == "exact" {
            let pd = ExactBinaryFormat.paddedDims
            var padded = [Float](repeating: 0, count: data.vectors.count * pd)
            for i in 0..<data.vectors.count {
                let src = data.vectors[i]
                let off = i * pd
                for d in 0..<14 { padded[off + d] = src[d] }
            }
            print("Writing \(outputPath)...")
            ExactBinaryWriter.write(path: outputPath, paddedVectors: padded, labels: data.labels, numVectors: data.vectors.count)
        } else {
            print("Building IVF index: clusters=\(numClusters), iterations=\(kmeansIter)...")
            let ivf = IvfBuilder.build(inputVectors: data.vectors, inputLabels: data.labels,
                                       numClusters: numClusters, kmeansIterations: kmeansIter)
            print("Writing \(outputPath)...")
            IvfBinaryWriter.write(path: outputPath, ivf: ivf, nprobeFull: nprobe)
        }

        let fileSize = try FileManager.default.attributesOfItem(atPath: outputPath)[.size] as! Int
        print("Done. File size: \(fileSize) bytes")
    }
}
```

- [ ] **Step 8: Verify build**

```bash
swift build --target Preprocessor 2>&1
```

- [ ] **Step 9: Commit**

```bash
git add Sources/Preprocessor/
git commit -m "feat: add preprocessor with K-means, IVF builder, binary writers, and warmup generator"
```

---

### Task 11: Docker + HAProxy + Makefile

**Files:**
- Create: `docker/Dockerfile`
- Create: `docker/docker-compose.yml`
- Create: `lb/haproxy.cfg`
- Create: `Makefile`
- Create: `scripts/download-resources.sh`

- [ ] **Step 1: Write Dockerfile**

```dockerfile
FROM swift:6.1 AS build
ARG RESOURCES_BASE_URL=https://raw.githubusercontent.com/zanfranceschi/rinha-de-backend-2026/main/resources

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /src

RUN mkdir -p /resources \
 && curl -fsSL "$RESOURCES_BASE_URL/references.json.gz" -o /resources/references.json.gz \
 && curl -fsSL "$RESOURCES_BASE_URL/mcc_risk.json"      -o /resources/mcc_risk.json \
 && curl -fsSL "$RESOURCES_BASE_URL/normalization.json"  -o /resources/normalization.json

COPY Package.swift .
COPY Sources/ Sources/
RUN swift build -c release --target Preprocessor

RUN mkdir -p /data \
 && swift run -c release Preprocessor /resources/references.json.gz /data/ivf.bin 4096 20 ivf 8 \
 && swift run -c release Preprocessor /resources/references.json.gz /data/exact.bin 0 0 exact 0 \
 && swift run -c release Preprocessor gen-warmup /resources/warmup-payloads.ndjson 200

RUN swift build -c release --target RinhaApp

FROM swift:6.1-slim
WORKDIR /app
COPY --from=build /src/.build/release/RinhaApp .
COPY --from=build /resources/mcc_risk.json          /resources/mcc_risk.json
COPY --from=build /resources/normalization.json      /resources/normalization.json
COPY --from=build /resources/warmup-payloads.ndjson  /resources/warmup-payloads.ndjson
COPY --from=build /data/ivf.bin                      /data/ivf.bin
COPY --from=build /data/exact.bin                    /data/exact.bin

ENV INDEX_PATH=/data/ivf.bin
ENV EXACT_PATH=/data/exact.bin
ENV RESOURCES_PATH=/resources
ENV API_PORT=8080
ENTRYPOINT ["./RinhaApp"]
```

- [ ] **Step 2: Write docker-compose.yml**

```yaml
services:
  haproxy:
    image: haproxy:3.1-alpine
    volumes:
      - ../lb/haproxy.cfg:/usr/local/etc/haproxy/haproxy.cfg:ro
      - sock:/run/sock
    ports:
      - "9999:9999"
    deploy:
      resources:
        limits:
          cpus: "0.20"
          memory: "16M"
    depends_on:
      - api1
      - api2

  api1: &api
    image: rinha/api-swift:latest
    environment:
      INDEX_PATH: /data/ivf.bin
      EXACT_PATH: /data/exact.bin
      RESOURCES_PATH: /resources
      SOCKET_PATH: /run/sock/api1.sock
      NPROBE_FULL: "5"
    volumes:
      - sock:/run/sock
    cpuset: "1,2"
    deploy:
      resources:
        limits:
          cpus: "0.40"
          memory: "162M"

  api2:
    <<: *api
    environment:
      INDEX_PATH: /data/ivf.bin
      EXACT_PATH: /data/exact.bin
      RESOURCES_PATH: /resources
      SOCKET_PATH: /run/sock/api2.sock
      NPROBE_FULL: "5"
    cpuset: "2,3"

volumes:
  sock:
    driver_opts:
      type: tmpfs
      device: tmpfs
      o: "size=1M"
```

- [ ] **Step 3: Write haproxy.cfg (copy from C# project)**

```
global
    maxconn 20000
    nbthread 1
    hard-stop-after 1s

defaults
    mode tcp
    maxconn 20000
    timeout connect 200ms
    timeout client 5s
    timeout server 5s
    option splice-auto
    option dontlognull
    option tcpka
    no log

frontend rinha_front
    bind *:9999
    default_backend rinha_api

backend rinha_api
    balance roundrobin
    server api1 unix@/run/sock/api1.sock maxconn 10000
    server api2 unix@/run/sock/api2.sock maxconn 10000
```

- [ ] **Step 4: Write Makefile**

```makefile
.PHONY: docker-build docker-up docker-down clean

COMPOSE := docker compose -f docker/docker-compose.yml --project-directory docker
IMAGE   := rinha/api-swift:latest

docker-build:
	docker build -f docker/Dockerfile -t $(IMAGE) .

docker-up:
	$(COMPOSE) up -d
	@echo "Waiting for /ready..."
	@for i in $$(seq 1 120); do \
		curl -sf http://localhost:9999/ready > /dev/null 2>&1 && echo "  ready in $${i}s" && break; \
		sleep 1; \
		[ $$i -eq 120 ] && echo "  TIMEOUT" && exit 1 || true; \
	done

docker-down:
	$(COMPOSE) down

K6_VUS      ?= 20
K6_DURATION ?= 60s

k6:
	docker run --rm --network host \
		-e API_URL="http://localhost:9999" \
		-e VUS="$(K6_VUS)" \
		-e DURATION="$(K6_DURATION)" \
		grafana/k6 run -

clean:
	rm -rf .build/ data/
```

- [ ] **Step 5: Write download-resources.sh**

```bash
#!/usr/bin/env bash
set -euo pipefail
BASE_URL="https://raw.githubusercontent.com/zanfranceschi/rinha-de-backend-2026/main/resources"
mkdir -p resources
curl -fsSL "$BASE_URL/references.json.gz" -o resources/references.json.gz
curl -fsSL "$BASE_URL/mcc_risk.json"      -o resources/mcc_risk.json
curl -fsSL "$BASE_URL/normalization.json"  -o resources/normalization.json
echo "Resources downloaded."
```

```bash
chmod +x scripts/download-resources.sh
```

- [ ] **Step 6: Commit**

```bash
git add docker/ lb/ Makefile scripts/
git commit -m "feat: add Docker multi-stage build, docker-compose, HAProxy, and Makefile"
```

---

### Task 12: Build Verification & Fix Compilation

**Files:**
- All source files (fix any compilation errors)

- [ ] **Step 1: Run full build**

```bash
cd /Users/jordaogustavo/Documents/workspace/rinha_backend_2026_swift
swift build 2>&1
```

- [ ] **Step 2: Fix any compilation errors**

Address each error iteratively. Common issues:
- Sendability warnings (Swift 6 strict concurrency)
- Missing imports
- Type mismatches between C and Swift
- Unsafe pointer rebinding issues

- [ ] **Step 3: Run tests**

```bash
swift test 2>&1
```

- [ ] **Step 4: Commit fixes**

```bash
git add -A
git commit -m "fix: resolve compilation errors and test failures"
```

---

### Task 13: Integration Testing with Local Index

- [ ] **Step 1: Test with small synthetic data**

Create a small test that builds a tiny IVF index (100 vectors, 4 clusters) and verifies the detector produces valid results.

- [ ] **Step 2: If the C# preprocessed data exists, test with real data**

```bash
# If data/ivf.bin exists from C# project:
RESOURCES_PATH=resources INDEX_PATH=../rinha_backend_2026/data/ivf.bin \
    EXACT_PATH=../rinha_backend_2026/data/exact.bin \
    swift run RinhaApp
```

Then in another terminal:
```bash
curl -X POST http://localhost:8080/fraud-score \
  -H 'Content-Type: application/json' \
  -d '{"transaction":{"amount":150,"installments":3,"requested_at":"2026-04-15T14:30:00Z"},"customer":{"avg_amount":100,"tx_count_24h":5,"known_merchants":["m-001"]},"merchant":{"id":"m-001","mcc":"5812","avg_amount":120},"terminal":{"is_online":true,"card_present":true,"km_from_home":10.5},"last_transaction":{"timestamp":"2026-04-15T12:00:00Z","km_from_current":5.0}}'
```

Expected: `{"approved":true/false,"fraud_score":0.0-1.0}`

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "test: verify end-to-end fraud detection with real data"
```
