import Foundation
import Dispatch

enum KMeans {
    /// Run K-means clustering.
    /// - Parameters:
    ///   - flatVectors: n * paddedDims flat float array
    ///   - n: number of vectors
    ///   - dims: padded dimensionality (16)
    ///   - k: number of clusters
    ///   - iterations: number of K-means iterations
    /// - Returns: array of n cluster assignments
    static func cluster(flatVectors: [Float], n: Int, dims: Int, k: Int, iterations: Int) -> [Int] {
        precondition(flatVectors.count == n * dims)
        precondition(k <= n)

        // Initialize centroids: evenly spaced from input
        var centroids = [Float](repeating: 0, count: k * dims)
        for c in 0..<k {
            let srcIdx = c * n / k
            let srcOff = srcIdx * dims
            let dstOff = c * dims
            for d in 0..<dims {
                centroids[dstOff + d] = flatVectors[srcOff + d]
            }
        }

        var assignments = [Int](repeating: 0, count: n)

        let numThreads = ProcessInfo.processInfo.activeProcessorCount
        let chunkSize = (n + numThreads - 1) / numThreads

        for iter in 0..<iterations {
            let iterStart = CFAbsoluteTimeGetCurrent()

            // Assignment step (parallel)
            DispatchQueue.concurrentPerform(iterations: numThreads) { threadIdx in
                let start = threadIdx * chunkSize
                let end = min(start + chunkSize, n)

                for i in start..<end {
                    let vecOff = i * dims
                    var bestDist = Float.infinity
                    var bestCluster = 0

                    for c in 0..<k {
                        let centOff = c * dims
                        var dist: Float = 0
                        for d in 0..<dims {
                            let diff = flatVectors[vecOff + d] - centroids[centOff + d]
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

            // Update step: recompute centroids
            // Use per-thread accumulators to avoid contention
            let perThreadSums = UnsafeMutableBufferPointer<Float>.allocate(capacity: numThreads * k * dims)
            perThreadSums.initialize(repeating: 0)
            let perThreadCounts = UnsafeMutableBufferPointer<Int>.allocate(capacity: numThreads * k)
            perThreadCounts.initialize(repeating: 0)

            DispatchQueue.concurrentPerform(iterations: numThreads) { threadIdx in
                let start = threadIdx * chunkSize
                let end = min(start + chunkSize, n)
                let sumBase = threadIdx * k * dims
                let countBase = threadIdx * k

                for i in start..<end {
                    let c = assignments[i]
                    let vecOff = i * dims
                    let dstOff = sumBase + c * dims
                    perThreadCounts[countBase + c] += 1
                    for d in 0..<dims {
                        perThreadSums[dstOff + d] += flatVectors[vecOff + d]
                    }
                }
            }

            // Merge accumulators
            var sums = [Float](repeating: 0, count: k * dims)
            var counts = [Int](repeating: 0, count: k)

            for t in 0..<numThreads {
                let sumBase = t * k * dims
                let countBase = t * k
                for c in 0..<k {
                    counts[c] += perThreadCounts[countBase + c]
                    let srcOff = sumBase + c * dims
                    let dstOff = c * dims
                    for d in 0..<dims {
                        sums[dstOff + d] += perThreadSums[srcOff + d]
                    }
                }
            }

            perThreadSums.deallocate()
            perThreadCounts.deallocate()

            // Compute new centroids
            for c in 0..<k {
                let off = c * dims
                if counts[c] > 0 {
                    let invCount = 1.0 / Float(counts[c])
                    for d in 0..<dims {
                        centroids[off + d] = sums[off + d] * invCount
                    }
                }
                // If count == 0, keep the old centroid (already in centroids)
            }

            let iterEnd = CFAbsoluteTimeGetCurrent()
            print("  K-means iteration \(iter + 1)/\(iterations) completed in \(String(format: "%.2f", iterEnd - iterStart))s")
        }

        // Final assignment pass
        DispatchQueue.concurrentPerform(iterations: numThreads) { threadIdx in
            let start = threadIdx * chunkSize
            let end = min(start + chunkSize, n)

            for i in start..<end {
                let vecOff = i * dims
                var bestDist = Float.infinity
                var bestCluster = 0

                for c in 0..<k {
                    let centOff = c * dims
                    var dist: Float = 0
                    for d in 0..<dims {
                        let diff = flatVectors[vecOff + d] - centroids[centOff + d]
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

        return assignments
    }
}
