import Foundation
import Dispatch

enum KMeans {
    static func cluster(flatVectors: [Float], n: Int, dims: Int, k: Int, iterations: Int) -> [Int] {
        precondition(flatVectors.count == n * dims)
        precondition(k <= n)

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
            let iterStart = Date()

            // Assignment step (parallel)
            assignments.withUnsafeMutableBufferPointer { assignBuf in
                let assignPtr = assignBuf.baseAddress!
                centroids.withUnsafeBufferPointer { centBuf in
                    let centPtr = centBuf.baseAddress!
                    flatVectors.withUnsafeBufferPointer { vecBuf in
                        let vecPtr = vecBuf.baseAddress!
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
                                        let diff = vecPtr[vecOff + d] - centPtr[centOff + d]
                                        dist += diff * diff
                                    }
                                    if dist < bestDist {
                                        bestDist = dist
                                        bestCluster = c
                                    }
                                }
                                assignPtr[i] = bestCluster
                            }
                        }
                    }
                }
            }

            // Update step: per-thread accumulators
            let perThreadSums = UnsafeMutablePointer<Float>.allocate(capacity: numThreads * k * dims)
            perThreadSums.initialize(repeating: 0, count: numThreads * k * dims)
            let perThreadCounts = UnsafeMutablePointer<Int>.allocate(capacity: numThreads * k)
            perThreadCounts.initialize(repeating: 0, count: numThreads * k)

            assignments.withUnsafeBufferPointer { assignBuf in
                let assignPtr = assignBuf.baseAddress!
                flatVectors.withUnsafeBufferPointer { vecBuf in
                    let vecPtr = vecBuf.baseAddress!
                    DispatchQueue.concurrentPerform(iterations: numThreads) { threadIdx in
                        let start = threadIdx * chunkSize
                        let end = min(start + chunkSize, n)
                        let sumBase = threadIdx * k * dims
                        let countBase = threadIdx * k

                        for i in start..<end {
                            let c = assignPtr[i]
                            let vecOff = i * dims
                            let dstOff = sumBase + c * dims
                            perThreadCounts[countBase + c] += 1
                            for d in 0..<dims {
                                perThreadSums[dstOff + d] += vecPtr[vecOff + d]
                            }
                        }
                    }
                }
            }

            // Merge accumulators
            var counts = [Int](repeating: 0, count: k)
            var sums = [Float](repeating: 0, count: k * dims)

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

            for c in 0..<k {
                let off = c * dims
                if counts[c] > 0 {
                    let invCount = 1.0 / Float(counts[c])
                    for d in 0..<dims {
                        centroids[off + d] = sums[off + d] * invCount
                    }
                }
            }

            let elapsed = Date().timeIntervalSince(iterStart)
            print("  K-means iteration \(iter + 1)/\(iterations) completed in \(String(format: "%.2f", elapsed))s")
        }

        // Final assignment pass
        assignments.withUnsafeMutableBufferPointer { assignBuf in
            let assignPtr = assignBuf.baseAddress!
            centroids.withUnsafeBufferPointer { centBuf in
                let centPtr = centBuf.baseAddress!
                flatVectors.withUnsafeBufferPointer { vecBuf in
                    let vecPtr = vecBuf.baseAddress!
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
                                    let diff = vecPtr[vecOff + d] - centPtr[centOff + d]
                                    dist += diff * diff
                                }
                                if dist < bestDist {
                                    bestDist = dist
                                    bestCluster = c
                                }
                            }
                            assignPtr[i] = bestCluster
                        }
                    }
                }
            }
        }

        return assignments
    }
}
