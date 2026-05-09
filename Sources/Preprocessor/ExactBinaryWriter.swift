import Foundation
import FraudDetector

enum ExactBinaryWriter {
    static func write(vectors: [Float], labels: [UInt8], numVectors: Int, to path: String) throws {
        let pd = ExactBinaryFormat.paddedDims

        precondition(vectors.count == numVectors * pd)
        precondition(labels.count == numVectors)

        FileManager.default.createFile(atPath: path, contents: nil)
        guard let fh = FileHandle(forWritingAtPath: path) else {
            throw PreprocessorError.writeFailed("Cannot open \(path) for writing")
        }
        defer { fh.closeFile() }

        // Write header (32 bytes)
        var header = Data(count: ExactBinaryFormat.headerSize)
        header.withUnsafeMutableBytes { ptr in
            let base = ptr.baseAddress!

            // magic (4 bytes)
            let magic = ExactBinaryFormat.magic
            base.storeBytes(of: magic[0], toByteOffset: 0, as: UInt8.self)
            base.storeBytes(of: magic[1], toByteOffset: 1, as: UInt8.self)
            base.storeBytes(of: magic[2], toByteOffset: 2, as: UInt8.self)
            base.storeBytes(of: magic[3], toByteOffset: 3, as: UInt8.self)

            // version (4 bytes)
            base.storeBytes(of: ExactBinaryFormat.version, toByteOffset: 4, as: UInt32.self)

            // numVectors (4 bytes)
            base.storeBytes(of: UInt32(numVectors), toByteOffset: 8, as: UInt32.self)

            // remaining bytes (12-31) are zero padding
        }
        fh.write(header)

        // Write float32 vectors (n * 16 * 4 bytes)
        fh.write(vectors.withUnsafeBytes { Data($0) })

        // Write labels (n bytes)
        fh.write(labels.withUnsafeBytes { Data($0) })

        let fileSize = fh.offsetInFile
        print("  Written \(fileSize / 1_048_576) MB to \(path)")
    }
}
