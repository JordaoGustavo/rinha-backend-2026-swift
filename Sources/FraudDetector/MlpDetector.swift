import Foundation

public final class MlpDetector: @unchecked Sendable {
    private let numLayers: Int
    private let threshold: Float
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

    public struct MlpResult {
        public let confident: Bool
        public let isFraud: Bool
        public let probability: Float
    }

    public func predict(vector: UnsafePointer<Float>, dims: Int) -> MlpResult {
        var input = [Float](repeating: 0, count: dims)
        for i in 0..<dims { input[i] = vector[i] }

        var current = input

        for layer in 0..<numLayers {
            let (inSize, outSize) = layerSizes[layer]
            let w = weights[layer]
            let b = biases[layer]
            var output = [Float](repeating: 0, count: outSize)

            for j in 0..<outSize {
                var sum = b[j]
                let wOff = j * inSize
                for i in 0..<inSize {
                    sum += current[i] * w[wOff + i]
                }
                if layer < numLayers - 1 {
                    output[j] = sum > 0 ? sum : 0  // ReLU
                } else {
                    output[j] = 1.0 / (1.0 + expf(-sum))  // Sigmoid
                }
            }
            current = output
        }

        let prob = current[0]
        let confidence = abs(prob - 0.5) * 2.0
        let isFraud = prob >= 0.5

        return MlpResult(
            confident: confidence >= threshold,
            isFraud: isFraud,
            probability: prob
        )
    }
}
