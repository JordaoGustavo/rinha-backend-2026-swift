#if canImport(Glibc)
import Glibc
#elseif canImport(Musl)
import Musl
#elseif canImport(Darwin)
import Darwin
#endif

import Foundation

public final class ExactDetector: @unchecked Sendable {
    private let fd: Int32
    private let basePtr: UnsafeMutableRawPointer
    private let mapSize: Int
    private let vectors: UnsafePointer<Float>
    private let labels: UnsafePointer<UInt8>
    private let numVectors: Int
    private let paddedDims: Int

    public init(path: String) throws {
        paddedDims = ExactBinaryFormat.paddedDims

        fd = open(path, O_RDONLY)
        guard fd >= 0 else {
            throw NSError(domain: "ExactDetector", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Failed to open \(path): errno \(errno)"])
        }

        // Get file size
        var st = stat()
        guard fstat(fd, &st) == 0 else {
            close(fd)
            throw NSError(domain: "ExactDetector", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "Failed to stat \(path): errno \(errno)"])
        }
        mapSize = Int(st.st_size)

        // mmap the file
        let ptr = mmap(nil, mapSize, PROT_READ, MAP_PRIVATE, fd, 0)
        guard ptr != MAP_FAILED else {
            close(fd)
            throw NSError(domain: "ExactDetector", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "Failed to mmap \(path): errno \(errno)"])
        }
        basePtr = ptr!

        // Validate magic
        let magic = basePtr.assumingMemoryBound(to: UInt8.self)
        guard magic[0] == ExactBinaryFormat.magic[0],
              magic[1] == ExactBinaryFormat.magic[1],
              magic[2] == ExactBinaryFormat.magic[2],
              magic[3] == ExactBinaryFormat.magic[3] else {
            munmap(basePtr, mapSize)
            close(fd)
            throw NSError(domain: "ExactDetector", code: 4,
                          userInfo: [NSLocalizedDescriptionKey: "Invalid magic in \(path)"])
        }

        // Read version (offset 4) and numVectors (offset 8)
        let version = basePtr.load(fromByteOffset: 4, as: UInt32.self)
        guard version == ExactBinaryFormat.version else {
            munmap(basePtr, mapSize)
            close(fd)
            throw NSError(domain: "ExactDetector", code: 5,
                          userInfo: [NSLocalizedDescriptionKey: "Unsupported version \(version) in \(path)"])
        }

        numVectors = Int(basePtr.load(fromByteOffset: 8, as: UInt32.self))

        // Set up pointers
        vectors = UnsafePointer((basePtr + ExactBinaryFormat.vectorsOffset).assumingMemoryBound(to: Float.self))
        labels = UnsafePointer((basePtr + ExactBinaryFormat.labelsOffset(numVectors)).assumingMemoryBound(to: UInt8.self))

        // Apply mmap hints
        MmapHints.hintWillNeed(basePtr, mapSize)
        MmapHints.hintRandom(basePtr, mapSize)
    }

    deinit {
        munmap(basePtr, mapSize)
        close(fd)
    }

    /// Walk every 4KB page to fault them in.
    public func prefault() {
        let pageSize = 4096
        var sum: UInt8 = 0
        let bytePtr = basePtr.assumingMemoryBound(to: UInt8.self)
        var offset = 0
        while offset < mapSize {
            sum &+= bytePtr[offset]
            offset += pageSize
        }
        // Prevent the compiler from optimizing away the loop
        withExtendedLifetime(sum) {}
    }

    /// Find top-5 nearest neighbors by float32 L2 squared distance.
    /// Returns (approved: fraudCount < 3, fraudCount).
    public func score(_ query: UnsafePointer<Float>) -> (approved: Bool, fraudCount: Int) {
        // Max-heap of (distance, index) with capacity 5
        // We track the 5 smallest distances using a max-heap so we can
        // quickly reject candidates larger than the current worst.
        var heapDist = (Float.infinity, Float.infinity, Float.infinity, Float.infinity, Float.infinity)
        var heapIdx = (-1, -1, -1, -1, -1)
        var heapCount = 0

        for i in 0..<numVectors {
            let vecPtr = vectors.advanced(by: i * paddedDims)
            let dist = SimdDistance.float32L2Squared(query, vecPtr)

            if heapCount < 5 {
                // Insert into heap
                insertMaxHeap(dist: dist, idx: i, heapDist: &heapDist, heapIdx: &heapIdx, count: &heapCount)
            } else if dist < heapDist.0 {
                // Replace root (max element) and sift down
                heapDist.0 = dist
                heapIdx.0 = i
                siftDown(heapDist: &heapDist, heapIdx: &heapIdx)
            }
        }

        // Count fraud labels among top-5
        var fraudCount = 0
        let indices = [heapIdx.0, heapIdx.1, heapIdx.2, heapIdx.3, heapIdx.4]
        for j in 0..<heapCount {
            let idx = indices[j]
            if idx >= 0 && labels[idx] != 0 {
                fraudCount += 1
            }
        }

        return (approved: fraudCount < 3, fraudCount: fraudCount)
    }

    // MARK: - Max-Heap helpers (capacity 5, stored in tuple for stack allocation)

    @inline(__always)
    private func insertMaxHeap(dist: Float, idx: Int,
                               heapDist: inout (Float, Float, Float, Float, Float),
                               heapIdx: inout (Int, Int, Int, Int, Int),
                               count: inout Int) {
        // Place at end and sift up
        setHeap(heapDist: &heapDist, heapIdx: &heapIdx, at: count, dist: dist, idx: idx)
        count += 1
        siftUp(heapDist: &heapDist, heapIdx: &heapIdx, index: count - 1)
    }

    @inline(__always)
    private func getHeapDist(_ h: (Float, Float, Float, Float, Float), _ i: Int) -> Float {
        switch i {
        case 0: return h.0
        case 1: return h.1
        case 2: return h.2
        case 3: return h.3
        case 4: return h.4
        default: return Float.infinity
        }
    }

    @inline(__always)
    private func getHeapIdx(_ h: (Int, Int, Int, Int, Int), _ i: Int) -> Int {
        switch i {
        case 0: return h.0
        case 1: return h.1
        case 2: return h.2
        case 3: return h.3
        case 4: return h.4
        default: return -1
        }
    }

    @inline(__always)
    private func setHeap(heapDist: inout (Float, Float, Float, Float, Float),
                         heapIdx: inout (Int, Int, Int, Int, Int),
                         at i: Int, dist: Float, idx: Int) {
        switch i {
        case 0: heapDist.0 = dist; heapIdx.0 = idx
        case 1: heapDist.1 = dist; heapIdx.1 = idx
        case 2: heapDist.2 = dist; heapIdx.2 = idx
        case 3: heapDist.3 = dist; heapIdx.3 = idx
        case 4: heapDist.4 = dist; heapIdx.4 = idx
        default: break
        }
    }

    @inline(__always)
    private func swap(heapDist: inout (Float, Float, Float, Float, Float),
                      heapIdx: inout (Int, Int, Int, Int, Int),
                      _ a: Int, _ b: Int) {
        let tmpD = getHeapDist(heapDist, a)
        let tmpI = getHeapIdx(heapIdx, a)
        setHeap(heapDist: &heapDist, heapIdx: &heapIdx, at: a,
                dist: getHeapDist(heapDist, b), idx: getHeapIdx(heapIdx, b))
        setHeap(heapDist: &heapDist, heapIdx: &heapIdx, at: b,
                dist: tmpD, idx: tmpI)
    }

    @inline(__always)
    private func siftUp(heapDist: inout (Float, Float, Float, Float, Float),
                        heapIdx: inout (Int, Int, Int, Int, Int),
                        index: Int) {
        var i = index
        while i > 0 {
            let parent = (i - 1) / 2
            if getHeapDist(heapDist, i) > getHeapDist(heapDist, parent) {
                swap(heapDist: &heapDist, heapIdx: &heapIdx, i, parent)
                i = parent
            } else {
                break
            }
        }
    }

    @inline(__always)
    private func siftDown(heapDist: inout (Float, Float, Float, Float, Float),
                          heapIdx: inout (Int, Int, Int, Int, Int)) {
        var i = 0
        while true {
            let left = 2 * i + 1
            let right = 2 * i + 2
            var largest = i

            if left < 5 && getHeapDist(heapDist, left) > getHeapDist(heapDist, largest) {
                largest = left
            }
            if right < 5 && getHeapDist(heapDist, right) > getHeapDist(heapDist, largest) {
                largest = right
            }

            if largest != i {
                swap(heapDist: &heapDist, heapIdx: &heapIdx, i, largest)
                i = largest
            } else {
                break
            }
        }
    }
}
