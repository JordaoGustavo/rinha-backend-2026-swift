import Foundation
import FraudDetector

enum IvfBinaryWriter {
    static func write(result: IvfResult, nprobeFast: Int, nprobeFull: Int, to path: String) throws {
        let k = result.numClusters
        let n = result.numVectors
        let pd = IvfBinaryFormat.paddedDims
        let bv = IvfBinaryFormat.blockVectors
        let scale = Float(IvfBinaryFormat.scale)

        // Step 1: Compute padded slot counts (round up each cluster to multiple of blockVectors)
        var paddedCounts = [Int](repeating: 0, count: k)
        var totalSlots = 0
        for c in 0..<k {
            let count = result.clusterCounts[c]
            let padded = ((count + bv - 1) / bv) * bv
            paddedCounts[c] = padded
            totalSlots += padded
        }

        print("  Total slots (padded): \(totalSlots) (from \(n) vectors)")

        // Step 2: Quantize vectors to Int16
        var int16Vectors = [Int16](repeating: 0, count: totalSlots * pd)
        var slotOffset = 0
        for c in 0..<k {
            let srcStart = result.clusterOffsets[c]
            let count = result.clusterCounts[c]
            let padded = paddedCounts[c]

            for j in 0..<padded {
                let dstOff = (slotOffset + j) * pd
                if j < count {
                    let srcOff = (srcStart + j) * pd
                    for d in 0..<pd {
                        let v = result.vectors[srcOff + d] * scale
                        let clamped = max(Float(Int16.min), min(Float(Int16.max), v))
                        int16Vectors[dstOff + d] = Int16(clamped)
                    }
                }
                // else: already zero-initialized (padding)
            }
            slotOffset += padded
        }

        // Step 3: Compute bounding boxes per cluster (min/max Int16 per dimension)
        var bboxMin = [Int16](repeating: Int16.max, count: k * pd)
        var bboxMax = [Int16](repeating: Int16.min, count: k * pd)

        slotOffset = 0
        for c in 0..<k {
            let count = result.clusterCounts[c]
            let padded = paddedCounts[c]
            let bbOff = c * pd

            for j in 0..<count {
                let vecOff = (slotOffset + j) * pd
                for d in 0..<pd {
                    let v = int16Vectors[vecOff + d]
                    if v < bboxMin[bbOff + d] { bboxMin[bbOff + d] = v }
                    if v > bboxMax[bbOff + d] { bboxMax[bbOff + d] = v }
                }
            }

            // Handle empty clusters
            if count == 0 {
                for d in 0..<pd {
                    bboxMin[bbOff + d] = 0
                    bboxMax[bbOff + d] = 0
                }
            }

            slotOffset += padded
        }

        // Step 4: SoA block layout
        // For block b, dim-pair kp, vector v within block:
        //   blockedVectors[b * pd * bv + kp * 2 * bv + v * 2 + {0,1}] = int16Vectors[(blockStart + v) * pd + 2*kp + {0,1}]
        var blockedVectors = [Int16](repeating: 0, count: totalSlots * pd)

        let numDimPairs = pd / 2  // 8 pairs for 16 dims
        slotOffset = 0
        for c in 0..<k {
            let padded = paddedCounts[c]
            let numBlocks = padded / bv

            for bLocal in 0..<numBlocks {
                let blockStart = slotOffset + bLocal * bv
                let b = blockStart / bv  // global block index
                let blockBase = b * pd * bv

                for kp in 0..<numDimPairs {
                    for v in 0..<bv {
                        let srcIdx = (blockStart + v) * pd + 2 * kp
                        let dstIdx = blockBase + kp * 2 * bv + v * 2
                        blockedVectors[dstIdx] = int16Vectors[srcIdx]
                        blockedVectors[dstIdx + 1] = int16Vectors[srcIdx + 1]
                    }
                }
            }

            slotOffset += padded
        }

        // Step 5: Quantize centroids to Int16
        var int16Centroids = [Int16](repeating: 0, count: k * pd)
        for c in 0..<k {
            let off = c * pd
            for d in 0..<pd {
                let v = result.centroids[off + d] * scale
                let clamped = max(Float(Int16.min), min(Float(Int16.max), v))
                int16Centroids[off + d] = Int16(clamped)
            }
        }

        // Step 6: Build cluster metadata
        var clusterMeta = [IvfBinaryFormat.ClusterMeta](repeating: .init(offset: 0, count: 0), count: k)
        var metaOffset: UInt32 = 0
        for c in 0..<k {
            clusterMeta[c] = IvfBinaryFormat.ClusterMeta(
                offset: metaOffset,
                count: UInt32(result.clusterCounts[c])
            )
            metaOffset += UInt32(paddedCounts[c])
        }

        // Step 7: Prepare padded labels and original indices (totalSlots, not n)
        var paddedLabels = [UInt8](repeating: 0, count: totalSlots)
        var paddedOriginalIndices = [Int32](repeating: -1, count: totalSlots)

        slotOffset = 0
        for c in 0..<k {
            let srcStart = result.clusterOffsets[c]
            let count = result.clusterCounts[c]
            let padded = paddedCounts[c]

            for j in 0..<count {
                paddedLabels[slotOffset + j] = result.labels[srcStart + j]
                paddedOriginalIndices[slotOffset + j] = result.originalIndices[srcStart + j]
            }
            slotOffset += padded
        }

        // Step 8: Write file
        print("  Writing IVF binary file...")

        FileManager.default.createFile(atPath: path, contents: nil)
        guard let fh = FileHandle(forWritingAtPath: path) else {
            throw PreprocessorError.writeFailed("Cannot open \(path) for writing")
        }
        defer { fh.closeFile() }

        // Write header (64 bytes)
        var header = Data(count: IvfBinaryFormat.headerSize)
        header.withUnsafeMutableBytes { ptr in
            let base = ptr.baseAddress!

            // magic (4 bytes)
            let magic = IvfBinaryFormat.magic
            base.storeBytes(of: magic[0], toByteOffset: 0, as: UInt8.self)
            base.storeBytes(of: magic[1], toByteOffset: 1, as: UInt8.self)
            base.storeBytes(of: magic[2], toByteOffset: 2, as: UInt8.self)
            base.storeBytes(of: magic[3], toByteOffset: 3, as: UInt8.self)

            // version (4 bytes)
            base.storeBytes(of: IvfBinaryFormat.version, toByteOffset: 4, as: UInt32.self)

            // numVectors (4 bytes)
            base.storeBytes(of: UInt32(n), toByteOffset: 8, as: UInt32.self)

            // numClusters (4 bytes)
            base.storeBytes(of: UInt32(k), toByteOffset: 12, as: UInt32.self)

            // dims (4 bytes)
            base.storeBytes(of: UInt32(IvfBinaryFormat.dims), toByteOffset: 16, as: UInt32.self)

            // paddedDims (4 bytes)
            base.storeBytes(of: UInt32(pd), toByteOffset: 20, as: UInt32.self)

            // nprobeFast (4 bytes)
            base.storeBytes(of: UInt32(nprobeFast), toByteOffset: 24, as: UInt32.self)

            // nprobeFull (4 bytes)
            base.storeBytes(of: UInt32(nprobeFull), toByteOffset: 28, as: UInt32.self)

            // scale (4 bytes)
            base.storeBytes(of: IvfBinaryFormat.scale, toByteOffset: 32, as: Int32.self)

            // totalSlots (4 bytes)
            base.storeBytes(of: UInt32(totalSlots), toByteOffset: 36, as: UInt32.self)

            // remaining bytes (40-63) are zero padding
        }
        fh.write(header)

        // Write centroids
        fh.write(dataFromArray(int16Centroids))

        // Write bboxMin
        fh.write(dataFromArray(bboxMin))

        // Write bboxMax
        fh.write(dataFromArray(bboxMax))

        // Write cluster metadata
        fh.write(dataFromArray(clusterMeta))

        // Write blocked vectors
        fh.write(dataFromArray(blockedVectors))

        // Write labels
        fh.write(dataFromArray(paddedLabels))

        // Write original indices
        fh.write(dataFromArray(paddedOriginalIndices))

        let fileSize = fh.offsetInFile
        print("  Written \(fileSize / 1_048_576) MB to \(path)")
    }

    private static func dataFromArray<T>(_ array: [T]) -> Data {
        return array.withUnsafeBytes { Data($0) }
    }
}
