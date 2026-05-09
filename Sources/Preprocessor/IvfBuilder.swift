import Foundation
import FraudDetector

public struct IvfResult {
    public var centroids: [Float]       // k * paddedDims
    public var vectors: [Float]         // n * paddedDims (reordered by cluster)
    public var labels: [UInt8]          // n (reordered)
    public var clusterOffsets: [Int]    // k
    public var clusterCounts: [Int]     // k
    public var originalIndices: [Int32] // n (maps clustered position -> original index)
    public var numVectors: Int
    public var numClusters: Int
}

enum IvfBuilder {
    static func build(vectors: [[Float]], labels: [UInt8], numClusters: Int,
                      kmeansIterations: Int) -> IvfResult {
        let n = vectors.count
        let dims = IvfBinaryFormat.dims
        let pd = IvfBinaryFormat.paddedDims
        let k = min(numClusters, n)

        // Step 1: Pad input vectors to paddedDims
        print("  Padding vectors to \(pd) dims...")
        var flat = [Float](repeating: 0, count: n * pd)
        for i in 0..<n {
            let src = vectors[i]
            let dstOff = i * pd
            for d in 0..<min(dims, src.count) {
                flat[dstOff + d] = src[d]
            }
        }

        // Step 2: Run K-means
        print("  Running K-means with k=\(k)...")
        let assignments = KMeans.cluster(flatVectors: flat, n: n, dims: pd, k: k,
                                         iterations: kmeansIterations)

        // Step 3: Compute cluster counts and offsets
        var clusterCounts = [Int](repeating: 0, count: k)
        for c in assignments {
            clusterCounts[c] += 1
        }

        var clusterOffsets = [Int](repeating: 0, count: k)
        var offset = 0
        for c in 0..<k {
            clusterOffsets[c] = offset
            offset += clusterCounts[c]
        }

        // Step 4: Compute centroids from assignments
        print("  Computing centroids...")
        var centroidSums = [Float](repeating: 0, count: k * pd)
        for i in 0..<n {
            let c = assignments[i]
            let srcOff = i * pd
            let dstOff = c * pd
            for d in 0..<pd {
                centroidSums[dstOff + d] += flat[srcOff + d]
            }
        }
        var centroids = [Float](repeating: 0, count: k * pd)
        for c in 0..<k {
            let off = c * pd
            if clusterCounts[c] > 0 {
                let inv = 1.0 / Float(clusterCounts[c])
                for d in 0..<pd {
                    centroids[off + d] = centroidSums[off + d] * inv
                }
            }
        }

        // Step 5: Reorder vectors by cluster
        print("  Reordering vectors by cluster...")
        var reorderedVectors = [Float](repeating: 0, count: n * pd)
        var reorderedLabels = [UInt8](repeating: 0, count: n)
        var originalIndices = [Int32](repeating: 0, count: n)

        // Track current write position per cluster
        var writePos = [Int](repeating: 0, count: k)
        for c in 0..<k {
            writePos[c] = clusterOffsets[c]
        }

        for i in 0..<n {
            let c = assignments[i]
            let pos = writePos[c]
            writePos[c] += 1

            let srcOff = i * pd
            let dstOff = pos * pd
            for d in 0..<pd {
                reorderedVectors[dstOff + d] = flat[srcOff + d]
            }
            reorderedLabels[pos] = labels[i]
            originalIndices[pos] = Int32(i)
        }

        // Step 6: Sort within each cluster by L2 distance to centroid (ascending)
        print("  Sorting within clusters by distance to centroid...")
        for c in 0..<k {
            let start = clusterOffsets[c]
            let count = clusterCounts[c]
            guard count > 1 else { continue }

            let centOff = c * pd

            // Compute distances
            var indexDist = [(index: Int, dist: Float)]()
            indexDist.reserveCapacity(count)

            for j in 0..<count {
                let pos = start + j
                let vecOff = pos * pd
                var dist: Float = 0
                for d in 0..<pd {
                    let diff = reorderedVectors[vecOff + d] - centroids[centOff + d]
                    dist += diff * diff
                }
                indexDist.append((index: j, dist: dist))
            }

            indexDist.sort { $0.dist < $1.dist }

            // Apply permutation
            var tmpVec = [Float](repeating: 0, count: count * pd)
            var tmpLbl = [UInt8](repeating: 0, count: count)
            var tmpIdx = [Int32](repeating: 0, count: count)

            for j in 0..<count {
                let srcPos = start + indexDist[j].index
                let srcOff = srcPos * pd
                let dstOff = j * pd
                for d in 0..<pd {
                    tmpVec[dstOff + d] = reorderedVectors[srcOff + d]
                }
                tmpLbl[j] = reorderedLabels[srcPos]
                tmpIdx[j] = originalIndices[srcPos]
            }

            // Copy back
            for j in 0..<count {
                let dstPos = start + j
                let dstOff = dstPos * pd
                let srcOff = j * pd
                for d in 0..<pd {
                    reorderedVectors[dstOff + d] = tmpVec[srcOff + d]
                }
                reorderedLabels[dstPos] = tmpLbl[j]
                originalIndices[dstPos] = tmpIdx[j]
            }
        }

        print("  IVF build complete: \(n) vectors, \(k) clusters")
        return IvfResult(
            centroids: centroids,
            vectors: reorderedVectors,
            labels: reorderedLabels,
            clusterOffsets: clusterOffsets,
            clusterCounts: clusterCounts,
            originalIndices: originalIndices,
            numVectors: n,
            numClusters: k
        )
    }
}
