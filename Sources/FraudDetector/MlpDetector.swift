import Foundation

public final class MlpDetector: @unchecked Sendable {
    private let numLayers: Int
    private var threshold: Float
    private let weights: [[Float]]  // transposed: cols × rows for fast matmul
    private let biases: [[Float]]
    private let layerSizes: [(Int, Int)]  // (input, output) per layer

    public init?(path: String) {
        guard FileManager.default.fileExists(atPath: path) else { return nil }
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)) else { return nil }

        var offset = 0
        func read<T>(_ type: T.Type) -> T {
            let size = MemoryLayout<T>.size
            let value = data.withUnsafeBytes { ptr in
                ptr.load(fromByteOffset: offset, as: T.self)
            }
            offset += size
            return value
        }

        let n = Int(read(UInt32.self))
        let thresh = read(Float.self)

        var ws = [[Float]]()
        var bs = [[Float]]()
        var sizes = [(Int, Int)]()

        for _ in 0..<n {
            let rows = Int(read(UInt32.self))
            let cols = Int(read(UInt32.self))
            sizes.append((rows, cols))

            var w = [Float](repeating: 0, count: cols * rows)
            data.withUnsafeBytes { ptr in
                let src = ptr.baseAddress!.advanced(by: offset).assumingMemoryBound(to: Float.self)
                for i in 0..<(cols * rows) { w[i] = src[i] }
            }
            offset += cols * rows * 4
            ws.append(w)

            var b = [Float](repeating: 0, count: cols)
            data.withUnsafeBytes { ptr in
                let src = ptr.baseAddress!.advanced(by: offset).assumingMemoryBound(to: Float.self)
                for i in 0..<cols { b[i] = src[i] }
            }
            offset += cols * 4
            bs.append(b)
        }

        self.numLayers = n
        self.threshold = thresh
        self.weights = ws
        self.biases = bs
        self.layerSizes = sizes
    }

    public func setThreshold(_ t: Float) { threshold = t }

    public struct MlpResult {
        public let confident: Bool
        public let isFraud: Bool
        public let probability: Float
    }

    public func predict(vector: UnsafePointer<Float>, dims: Int) -> MlpResult {
        let maxBuf = 512
        return withUnsafeTemporaryAllocation(of: Float.self, capacity: maxBuf) { buf1 in
            return withUnsafeTemporaryAllocation(of: Float.self, capacity: maxBuf) { buf2 in
                var src = buf1.baseAddress!
                var dst = buf2.baseAddress!

                for i in 0..<dims { src[i] = vector[i] }

                for layer in 0..<numLayers {
                    let (inSize, outSize) = layerSizes[layer]
                    let isLast = layer == numLayers - 1
                    weights[layer].withUnsafeBufferPointer { wBuf in
                        biases[layer].withUnsafeBufferPointer { bBuf in
                            let w = wBuf.baseAddress!
                            let b = bBuf.baseAddress!
                            for j in 0..<outSize {
                                let sum = b[j] + SimdDistance.float32Dot(src, w + j * inSize, inSize)
                                dst[j] = isLast ? 1.0 / (1.0 + expf(-sum)) : (sum > 0 ? sum : 0)
                            }
                        }
                    }
                    let tmp = src; src = dst; dst = tmp
                }

                let prob = src[0]
                let confidence = abs(prob - 0.5) * 2.0
                return MlpResult(confident: confidence >= threshold, isFraud: prob >= 0.5, probability: prob)
            }
        }
    }
}
