#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#elseif canImport(Musl)
import Musl
#endif

import Foundation

public final class IvfDetector: @unchecked Sendable {
    private let ivfBase: UnsafeMutableRawPointer
    private let ivfSize: Int
    private let ivfFd: Int32

    public let numVectors: Int
    public let numClusters: Int
    private let totalSlots: Int
    public let nprobeFull: Int

    private let centroids: UnsafePointer<Int16>
    private let bboxMin: UnsafePointer<Int16>
    private let bboxMax: UnsafePointer<Int16>
    private let clusterMeta: UnsafePointer<IvfBinaryFormat.ClusterMeta>
    private let vectors: UnsafePointer<Int16>
    private let labels: UnsafePointer<UInt8>
    private let originalIndices: UnsafePointer<Int32>

    private let exactBase: UnsafeMutableRawPointer?
    private let exactSize: Int
    private let exactFd: Int32
    private let exactVectors: UnsafePointer<Float>?
    private let exactNumVectors: Int

    private static let topK = 5
    private static let rerankK = 6
    private static let paddedDims = IvfBinaryFormat.paddedDims
    private static let blockVectors = IvfBinaryFormat.blockVectors

    public init(ivfPath: String, exactPath: String? = nil) throws {
        let fd = open(ivfPath, O_RDONLY)
        guard fd >= 0 else { throw IvfError.openFailed(ivfPath, errno) }

        var st = stat()
        guard fstat(fd, &st) == 0 else { close(fd); throw IvfError.statFailed(ivfPath, errno) }
        let fileSize = Int(st.st_size)
        guard fileSize >= IvfBinaryFormat.headerSize else { close(fd); throw IvfError.fileTooSmall(ivfPath) }

        guard let mapped = mmap(nil, fileSize, PROT_READ, MAP_PRIVATE, fd, 0),
              mapped != MAP_FAILED else { close(fd); throw IvfError.mmapFailed(ivfPath, errno) }

        self.ivfFd = fd
        self.ivfBase = mapped
        self.ivfSize = fileSize

        let headerPtr = mapped.assumingMemoryBound(to: UInt32.self)
        let version = headerPtr[1]
        guard version == IvfBinaryFormat.version else {
            throw IvfError.versionMismatch(Int(version), Int(IvfBinaryFormat.version))
        }

        self.numVectors = Int(headerPtr[2])
        self.numClusters = Int(headerPtr[3])
        self.totalSlots = Int(headerPtr[9])

        var nprobe = Int(headerPtr[7])
        if let envVal = ProcessInfo.processInfo.environment["NPROBE_FULL"],
           let override = Int(envVal), override > 0 {
            nprobe = override
        }
        self.nprobeFull = nprobe

        let base = mapped
        self.centroids = UnsafePointer(base.advanced(by: IvfBinaryFormat.centroidsOffset).assumingMemoryBound(to: Int16.self))
        self.bboxMin = UnsafePointer(base.advanced(by: IvfBinaryFormat.bboxMinOffset(numClusters)).assumingMemoryBound(to: Int16.self))
        self.bboxMax = UnsafePointer(base.advanced(by: IvfBinaryFormat.bboxMaxOffset(numClusters)).assumingMemoryBound(to: Int16.self))
        self.clusterMeta = UnsafePointer(base.advanced(by: IvfBinaryFormat.clusterMetaOffset(numClusters)).assumingMemoryBound(to: IvfBinaryFormat.ClusterMeta.self))
        self.vectors = UnsafePointer(base.advanced(by: IvfBinaryFormat.vectorsOffset(numClusters)).assumingMemoryBound(to: Int16.self))
        self.labels = UnsafePointer(base.advanced(by: IvfBinaryFormat.labelsOffset(numClusters, totalSlots)).assumingMemoryBound(to: UInt8.self))
        self.originalIndices = UnsafePointer(base.advanced(by: IvfBinaryFormat.originalIndicesOffset(numClusters, totalSlots)).assumingMemoryBound(to: Int32.self))

        if let exactPath = exactPath {
            let eFd = open(exactPath, O_RDONLY)
            guard eFd >= 0 else { throw IvfError.openFailed(exactPath, errno) }
            var eSt = stat()
            guard fstat(eFd, &eSt) == 0 else { close(eFd); throw IvfError.statFailed(exactPath, errno) }
            let eSize = Int(eSt.st_size)
            guard let eMapped = mmap(nil, eSize, PROT_READ, MAP_PRIVATE, eFd, 0),
                  eMapped != MAP_FAILED else { close(eFd); throw IvfError.mmapFailed(exactPath, errno) }

            let eHeader = eMapped.assumingMemoryBound(to: UInt32.self)
            let eVersion = eHeader[1]
            guard eVersion == ExactBinaryFormat.version else {
                throw IvfError.versionMismatch(Int(eVersion), Int(ExactBinaryFormat.version))
            }
            let eNumVectors = Int(eHeader[2])

            self.exactFd = eFd
            self.exactBase = eMapped
            self.exactSize = eSize
            self.exactNumVectors = eNumVectors
            self.exactVectors = UnsafePointer(eMapped.advanced(by: ExactBinaryFormat.vectorsOffset).assumingMemoryBound(to: Float.self))
        } else {
            self.exactFd = -1
            self.exactBase = nil
            self.exactSize = 0
            self.exactNumVectors = 0
            self.exactVectors = nil
        }

    }

    deinit {
        munmap(ivfBase, ivfSize)
        close(ivfFd)
        if let exactBase = exactBase {
            munmap(exactBase, exactSize)
            close(exactFd)
        }
    }

    public func prefault() {
        MmapHints.hintHugePages(ivfBase, ivfSize)
        MmapHints.hintWillNeed(ivfBase, ivfSize)
        if let exactBase = exactBase {
            MmapHints.hintHugePages(exactBase, exactSize)
            MmapHints.hintWillNeed(exactBase, exactSize)
        }
    }

    public struct ScoreResult {
        public let approved: Bool
        public let fraudCount: Int
    }

    public func score(_ queryFloat: UnsafePointer<Float>) -> ScoreResult {
        let pd = Self.paddedDims

        // Quantize float[16] -> int16[16]
        var queryQ = (
            Int16(0), Int16(0), Int16(0), Int16(0),
            Int16(0), Int16(0), Int16(0), Int16(0),
            Int16(0), Int16(0), Int16(0), Int16(0),
            Int16(0), Int16(0), Int16(0), Int16(0)
        )
        withUnsafeMutablePointer(to: &queryQ) { tuplePtr in
            let qPtr = UnsafeMutableRawPointer(tuplePtr).assumingMemoryBound(to: Int16.self)
            let scale = Float(IvfBinaryFormat.scale)
            for i in 0..<pd {
                let v = (queryFloat[i] * scale).rounded()
                let clamped = max(Float(Int16.min), min(Float(Int16.max), v))
                qPtr[i] = Int16(clamped)
            }
        }

        // Full IVF k-NN search
        var candidates = [(dist: Int32, slot: Int)](repeating: (Int32.max, -1), count: Self.rerankK)
        var candidateCount = 0

        withUnsafePointer(to: &queryQ) { tuplePtr in
            let qPtr = UnsafeRawPointer(tuplePtr).assumingMemoryBound(to: Int16.self)
            findKNearest(query: qPtr, results: &candidates, count: &candidateCount)
        }

        // Float32 rerank
        if let exactVecs = exactVectors {
            var rerankResults = [(dist: Float, label: UInt8)](repeating: (Float.greatestFiniteMagnitude, 0), count: Self.topK)
            var rerankCount = 0

            for i in 0..<candidateCount {
                let slot = candidates[i].slot
                guard slot >= 0 else { continue }
                let origIdx = Int(originalIndices[slot])
                guard origIdx >= 0, origIdx < exactNumVectors else { continue }

                let exactVec = exactVecs.advanced(by: origIdx * ExactBinaryFormat.paddedDims)
                let fDist = SimdDistance.float32L2Squared(queryFloat, exactVec)
                let label = labels[slot]

                if rerankCount < Self.topK || fDist < rerankResults[rerankCount - 1].dist {
                    let insertAt: Int
                    if rerankCount < Self.topK {
                        insertAt = rerankCount
                        rerankCount += 1
                    } else {
                        insertAt = Self.topK - 1
                    }
                    rerankResults[insertAt] = (fDist, label)
                    var j = insertAt
                    while j > 0, rerankResults[j].dist < rerankResults[j - 1].dist {
                        let tmp = rerankResults[j]
                        rerankResults[j] = rerankResults[j - 1]
                        rerankResults[j - 1] = tmp
                        j -= 1
                    }
                }
            }

            var fraudCount = 0
            for i in 0..<min(rerankCount, Self.topK) {
                if rerankResults[i].label == 1 { fraudCount += 1 }
            }
            return ScoreResult(approved: fraudCount < 3, fraudCount: fraudCount)
        } else {
            var fraudCount = 0
            for i in 0..<min(candidateCount, Self.topK) {
                let slot = candidates[i].slot
                guard slot >= 0 else { continue }
                if labels[slot] == 1 { fraudCount += 1 }
            }
            return ScoreResult(approved: fraudCount < 3, fraudCount: fraudCount)
        }
    }

    // MARK: - IVF search

    private struct HeapEntry {
        var dist: Int32
        var index: Int
    }

    private struct Candidate {
        var dist: Int32
        var slot: Int
    }

    private func findKNearest(
        query: UnsafePointer<Int16>,
        results: inout [(dist: Int32, slot: Int)],
        count: inout Int
    ) {
        let k = Self.rerankK

        // Phase 1: Centroid scan
        var centroidHeap = [HeapEntry](repeating: HeapEntry(dist: 0, index: 0), count: nprobeFull)
        var centroidHeapSize = 0
        let centroidStride = Self.paddedDims

        for c in 0..<numClusters {
            let centroidPtr = centroids.advanced(by: c * centroidStride)
            if c + 8 < numClusters {
                SimdDistance.prefetch(UnsafeRawPointer(centroids.advanced(by: (c + 8) * centroidStride)))
            }

            let dist = SimdDistance.int16L2Squared(query, centroidPtr)

            if centroidHeapSize < nprobeFull {
                centroidHeap[centroidHeapSize] = HeapEntry(dist: dist, index: c)
                centroidHeapSize += 1
                if centroidHeapSize == nprobeFull {
                    buildMaxHeap(&centroidHeap, centroidHeapSize)
                }
            } else if dist < centroidHeap[0].dist {
                centroidHeap[0] = HeapEntry(dist: dist, index: c)
                siftDown(&centroidHeap, 0, centroidHeapSize)
            }
        }

        var resultHeap = [Candidate](repeating: Candidate(dist: Int32.max, slot: -1), count: k)
        var resultHeapSize = 0

        // Scan probed clusters
        for i in 0..<centroidHeapSize {
            let clusterIdx = centroidHeap[i].index
            let meta = clusterMeta[clusterIdx]
            let clusterCount = Int(meta.count)
            guard clusterCount > 0 else { continue }

            scanCluster(
                query: query,
                clusterOffset: Int(meta.offset),
                clusterCount: clusterCount,
                heap: &resultHeap,
                heapSize: &resultHeapSize,
                k: k
            )
        }

        count = resultHeapSize
        for i in stride(from: resultHeapSize - 1, through: 0, by: -1) {
            results[i] = (resultHeap[0].dist, resultHeap[0].slot)
            resultHeap[0] = resultHeap[resultHeapSize - 1]
            resultHeapSize -= 1
            if resultHeapSize > 0 {
                siftDownCandidate(&resultHeap, 0, resultHeapSize)
            }
        }
    }

    // MARK: - ScanCluster with SIMD batch (8 vectors at once)

    @inline(__always)
    private func scanCluster(
        query: UnsafePointer<Int16>,
        clusterOffset: Int,
        clusterCount: Int,
        heap: inout [Candidate],
        heapSize: inout Int,
        k: Int
    ) {
        let blockSize = Self.blockVectors
        let pd = Self.paddedDims

        let fullBlocks = clusterCount / blockSize
        let remainder = clusterCount % blockSize

        var dists: (Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32) = (0,0,0,0,0,0,0,0)

        for b in 0..<fullBlocks {
            let blockBaseSlot = clusterOffset + b * blockSize
            let blockPtr = vectors.advanced(by: blockBaseSlot * pd)

            withUnsafeMutablePointer(to: &dists) { dp in
                SimdDistance.int16SoaBlockDistances(query, blockPtr,
                    UnsafeMutableRawPointer(dp).assumingMemoryBound(to: Int32.self))
            }

            withUnsafePointer(to: &dists) { dp in
                let d = UnsafeRawPointer(dp).assumingMemoryBound(to: Int32.self)
                for v in 0..<blockSize {
                    let dist = d[v]
                    let slot = blockBaseSlot + v
                    if heapSize < k {
                        heap[heapSize] = Candidate(dist: dist, slot: slot)
                        heapSize += 1
                        if heapSize == k { buildMaxHeapCandidate(&heap, heapSize) }
                    } else if dist < heap[0].dist {
                        heap[0] = Candidate(dist: dist, slot: slot)
                        siftDownCandidate(&heap, 0, heapSize)
                    }
                }
            }
        }

        if remainder > 0 {
            let blockBaseSlot = clusterOffset + fullBlocks * blockSize
            let blockPtr = vectors.advanced(by: blockBaseSlot * pd)

            withUnsafeMutablePointer(to: &dists) { dp in
                SimdDistance.int16SoaBlockDistances(query, blockPtr,
                    UnsafeMutableRawPointer(dp).assumingMemoryBound(to: Int32.self))
            }

            withUnsafePointer(to: &dists) { dp in
                let d = UnsafeRawPointer(dp).assumingMemoryBound(to: Int32.self)
                for v in 0..<remainder {
                    let dist = d[v]
                    let slot = blockBaseSlot + v
                    if heapSize < k {
                        heap[heapSize] = Candidate(dist: dist, slot: slot)
                        heapSize += 1
                        if heapSize == k { buildMaxHeapCandidate(&heap, heapSize) }
                    } else if dist < heap[0].dist {
                        heap[0] = Candidate(dist: dist, slot: slot)
                        siftDownCandidate(&heap, 0, heapSize)
                    }
                }
            }
        }
    }

    // MARK: - Heap helpers

    @inline(__always)
    private func buildMaxHeap(_ heap: inout [HeapEntry], _ n: Int) {
        var i = n / 2 - 1
        while i >= 0 { siftDown(&heap, i, n); i -= 1 }
    }

    @inline(__always)
    private func siftDown(_ heap: inout [HeapEntry], _ root: Int, _ n: Int) {
        var i = root
        while true {
            var largest = i
            let left = 2 * i + 1, right = 2 * i + 2
            if left < n, heap[left].dist > heap[largest].dist { largest = left }
            if right < n, heap[right].dist > heap[largest].dist { largest = right }
            if largest == i { break }
            let tmp = heap[i]; heap[i] = heap[largest]; heap[largest] = tmp
            i = largest
        }
    }

    @inline(__always)
    private func buildMaxHeapCandidate(_ heap: inout [Candidate], _ n: Int) {
        var i = n / 2 - 1
        while i >= 0 { siftDownCandidate(&heap, i, n); i -= 1 }
    }

    @inline(__always)
    private func siftDownCandidate(_ heap: inout [Candidate], _ root: Int, _ n: Int) {
        var i = root
        while true {
            var largest = i
            let left = 2 * i + 1, right = 2 * i + 2
            if left < n, heap[left].dist > heap[largest].dist { largest = left }
            if right < n, heap[right].dist > heap[largest].dist { largest = right }
            if largest == i { break }
            let tmp = heap[i]; heap[i] = heap[largest]; heap[largest] = tmp
            i = largest
        }
    }
}

public enum IvfError: Error, CustomStringConvertible {
    case openFailed(String, Int32)
    case statFailed(String, Int32)
    case fileTooSmall(String)
    case mmapFailed(String, Int32)
    case versionMismatch(Int, Int)

    public var description: String {
        switch self {
        case .openFailed(let path, let err): return "Failed to open '\(path)': errno=\(err)"
        case .statFailed(let path, let err): return "Failed to stat '\(path)': errno=\(err)"
        case .fileTooSmall(let path): return "File too small for IVF header: '\(path)'"
        case .mmapFailed(let path, let err): return "Failed to mmap '\(path)': errno=\(err)"
        case .versionMismatch(let got, let expected): return "Version mismatch: got \(got), expected \(expected)"
        }
    }
}
