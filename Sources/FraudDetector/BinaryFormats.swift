import Foundation

public enum IvfBinaryFormat {
    public static let magic: [UInt8] = [0x49, 0x56, 0x46, 0x52] // "IVFR"
    public static let version: UInt32 = 7
    public static let headerSize = 64
    public static let dims = 14
    public static let paddedDims = 16
    public static let scale: Int32 = 4096
    public static let blockVectors = 8
    public static let centroidVectorBytes = paddedDims * MemoryLayout<Int16>.size
    public static let int16VectorBytes = paddedDims * MemoryLayout<Int16>.size

    public struct ClusterMeta {
        public var offset: UInt32
        public var count: UInt32
        public init(offset: UInt32, count: UInt32) {
            self.offset = offset
            self.count = count
        }
    }

    public static var centroidsOffset: Int { headerSize }
    public static func bboxMinOffset(_ numClusters: Int) -> Int { centroidsOffset + numClusters * centroidVectorBytes }
    public static func bboxMaxOffset(_ numClusters: Int) -> Int { bboxMinOffset(numClusters) + numClusters * int16VectorBytes }
    public static func clusterMetaOffset(_ numClusters: Int) -> Int { bboxMaxOffset(numClusters) + numClusters * int16VectorBytes }
    public static func vectorsOffset(_ numClusters: Int) -> Int { clusterMetaOffset(numClusters) + numClusters * 8 }
    public static func labelsOffset(_ numClusters: Int, _ totalSlots: Int) -> Int { vectorsOffset(numClusters) + totalSlots * int16VectorBytes }
    public static func originalIndicesOffset(_ numClusters: Int, _ totalSlots: Int) -> Int { labelsOffset(numClusters, totalSlots) + totalSlots }
    public static func totalSize(_ numClusters: Int, _ totalSlots: Int) -> Int { originalIndicesOffset(numClusters, totalSlots) + totalSlots * MemoryLayout<Int32>.size }
}

public enum ExactBinaryFormat {
    public static let magic: [UInt8] = [0x45, 0x58, 0x43, 0x54] // "EXCT"
    public static let version: UInt32 = 1
    public static let headerSize = 32
    public static let dims = 14
    public static let paddedDims = 16
    public static let vectorBytes = paddedDims * MemoryLayout<Float>.size
    public static var vectorsOffset: Int { headerSize }
    public static func labelsOffset(_ numVectors: Int) -> Int { vectorsOffset + numVectors * vectorBytes }
    public static func totalSize(_ numVectors: Int) -> Int { labelsOffset(numVectors) + numVectors }
}
