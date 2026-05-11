public enum ProfileFastPath {
    public static let keyCount = 1 << 22 // 4,194,304 buckets
    public static let legitMask: UInt8 = 1
    public static let fraudMask: UInt8 = 2
    private static let installmentsThreshold: Int16 = 410 // 0.1 * 4096

    @inline(__always)
    public static func key(_ qInt: UnsafePointer<Int16>) -> Int {
        var k = 0
        k |= bucket16(qInt[5])
        k |= bucket8(qInt[7]) << 4
        k |= bucket4(qInt[10]) << 7
        k |= bucket4(qInt[11]) << 9
        k |= bucket4(qInt[8]) << 11
        k |= (qInt[3] < 0 ? 1 : 0) << 13
        k |= (qInt[2] > 0 ? 1 : 0) << 14
        k |= (qInt[1] > 0 ? 1 : 0) << 15
        k |= (qInt[4] > 0 ? 1 : 0) << 16
        k |= bucket4(qInt[0]) << 17
        k |= (qInt[9] > installmentsThreshold ? 1 : 0) << 19
        k |= bucket4(qInt[13]) << 20
        return k
    }

    @inline(__always)
    private static func bucket4(_ value: Int16) -> Int {
        if value <= 0 { return 0 }
        let b = Int(value) * 4 / (Int(IvfBinaryFormat.scale) + 1)
        return b > 3 ? 3 : b
    }

    @inline(__always)
    private static func bucket8(_ value: Int16) -> Int {
        if value <= 0 { return 0 }
        let b = Int(value) * 8 / (Int(IvfBinaryFormat.scale) + 1)
        return b > 7 ? 7 : b
    }

    @inline(__always)
    private static func bucket16(_ value: Int16) -> Int {
        if value <= 0 { return 0 }
        let b = Int(value) * 16 / (Int(IvfBinaryFormat.scale) + 1)
        return b > 15 ? 15 : b
    }

    public static func build(
        vectors: UnsafePointer<Int16>,
        labels: UnsafePointer<UInt8>,
        clusterMeta: UnsafePointer<IvfBinaryFormat.ClusterMeta>,
        numClusters: Int,
        paddedDims: Int,
        blockVectors: Int
    ) -> (mask: [UInt8], count: [UInt16]) {
        var mask = [UInt8](repeating: 0, count: keyCount)
        var count = [UInt16](repeating: 0, count: keyCount)

        var qBuf = [Int16](repeating: 0, count: paddedDims)

        for c in 0..<numClusters {
            let meta = clusterMeta[c]
            let clusterCount = Int(meta.count)
            let clusterOffset = Int(meta.offset)

            for i in 0..<clusterCount {
                let b = i / blockVectors
                let v = i % blockVectors
                let blockPtr = vectors.advanced(by: (clusterOffset + b * blockVectors) * paddedDims)

                for kp in 0..<(paddedDims / 2) {
                    qBuf[kp * 2]     = blockPtr[kp * 16 + v * 2]
                    qBuf[kp * 2 + 1] = blockPtr[kp * 16 + v * 2 + 1]
                }

                let slot = clusterOffset + i
                let label = labels[slot]
                let k = qBuf.withUnsafeBufferPointer { buf in
                    key(buf.baseAddress!)
                }

                if count[k] < UInt16.max { count[k] += 1 }
                mask[k] |= label == 1 ? fraudMask : legitMask
            }
        }

        return (mask, count)
    }
}
