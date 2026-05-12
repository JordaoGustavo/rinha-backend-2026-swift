import Foundation
import FraudDetector

let resourcesPath = ProcessInfo.processInfo.environment["RESOURCES_PATH"] ?? "/resources"
try MccRisk.initialize(
    mccRiskPath: resourcesPath + "/mcc_risk.json",
    normalizationPath: resourcesPath + "/normalization.json"
)

let dataPath = ProcessInfo.processInfo.environment["INDEX_PATH"] ?? "/data/ivf.bin"
let exactPath = ProcessInfo.processInfo.environment["EXACT_PATH"]
let modelPath = ProcessInfo.processInfo.environment["MODEL_PATH"] ?? resourcesPath + "/model.bin"
let socketPath = ProcessInfo.processInfo.environment["SOCKET_PATH"]
let port = Int(ProcessInfo.processInfo.environment["API_PORT"] ?? "8080") ?? 8080

print("Opening index from \(dataPath)...")
let detector: IvfDetector
if let ep = exactPath, FileManager.default.fileExists(atPath: ep) {
    print("  with exact float32 rerank from \(ep)")
    detector = try IvfDetector(ivfPath: dataPath, exactPath: ep)
} else {
    detector = try IvfDetector(ivfPath: dataPath)
}
print("Loaded \(detector.numVectors) vectors, \(detector.numClusters) clusters, nprobe=\(detector.nprobeFull)")

let mlp: MlpDetector?
if let m = MlpDetector(path: modelPath) {
    if let threshStr = ProcessInfo.processInfo.environment["MLP_THRESHOLD"],
       let t = Float(threshStr) {
        m.setThreshold(t)
        print("MLP model loaded from \(modelPath), threshold=\(t)")
    } else {
        print("MLP model loaded from \(modelPath)")
    }
    mlp = m
} else {
    mlp = nil
    print("No MLP model found, using k-NN only")
}

print("Prefaulting pages...")
detector.prefault()

let warmupIterations = 200
let warmupPath = resourcesPath + "/warmup-payloads.ndjson"
var warmedReal = 0
if FileManager.default.fileExists(atPath: warmupPath) {
    print("Warming up with real payloads from \(warmupPath)...")
    if let contents = try? String(contentsOfFile: warmupPath, encoding: .utf8) {
        var vector = [Float](repeating: 0, count: 16)
        for line in contents.split(separator: "\n") where warmedReal < warmupIterations {
            guard line.count > 4 else { continue }
            let data = Array(line.utf8)
            data.withUnsafeBufferPointer { buf in
                vector.withUnsafeMutableBufferPointer { vecBuf in
                    for i in 0..<16 { vecBuf[i] = 0 }
                    TransactionParser.parse(UnsafeRawBufferPointer(buf), into: vecBuf.baseAddress!)
                }
            }
            _ = vector.withUnsafeBufferPointer { buf in
                detector.score(buf.baseAddress!)
            }
            if let m = mlp {
                _ = vector.withUnsafeBufferPointer { buf in
                    m.predict(vector: buf.baseAddress!, dims: 14)
                }
            }
            warmedReal += 1
        }
    }
    print("  warmed up with \(warmedReal) real payloads.")
}
if warmedReal < warmupIterations {
    print("  filling remaining \(warmupIterations - warmedReal) with random vectors")
    var state: UInt64 = 42
    var vector = [Float](repeating: 0, count: 16)
    for _ in warmedReal..<warmupIterations {
        for d in 0..<14 {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            vector[d] = Float(state >> 33) / Float(1 << 31)
        }
        for d in 14..<16 { vector[d] = 0 }
        _ = vector.withUnsafeBufferPointer { buf in
            detector.score(buf.baseAddress!)
        }
        if let m = mlp {
            _ = vector.withUnsafeBufferPointer { buf in
                m.predict(vector: buf.baseAddress!, dims: 14)
            }
        }
    }
}
print("Ready.")

let server = RawServer(detector: detector, mlp: mlp, socketPath: socketPath, port: port)
server.run()
