import Foundation
import NIOCore
import NIOPosix
import NIOHTTP1
import FraudDetector

let resourcesPath = ProcessInfo.processInfo.environment["RESOURCES_PATH"] ?? "/resources"
try MccRisk.initialize(
    mccRiskPath: resourcesPath + "/mcc_risk.json",
    normalizationPath: resourcesPath + "/normalization.json"
)

let dataPath = ProcessInfo.processInfo.environment["INDEX_PATH"] ?? "/data/ivf.bin"
let exactPath = ProcessInfo.processInfo.environment["EXACT_PATH"]
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
    }
}
print("Ready.")

let group = MultiThreadedEventLoopGroup(numberOfThreads: System.coreCount)
defer { try? group.syncShutdownGracefully() }

let bootstrap = ServerBootstrap(group: group)
    .serverChannelOption(.backlog, value: 2048)
    .serverChannelOption(.socketOption(.so_reuseaddr), value: 1)
    .childChannelInitializer { channel in
        channel.pipeline.configureHTTPServerPipeline().flatMap {
            channel.pipeline.addHandler(HttpHandler(detector: detector))
        }
    }
    .childChannelOption(.socketOption(.so_reuseaddr), value: 1)
    .childChannelOption(.maxMessagesPerRead, value: 16)

let channel: Channel
if let sp = socketPath {
    if FileManager.default.fileExists(atPath: sp) {
        try FileManager.default.removeItem(atPath: sp)
    }
    channel = try bootstrap.bind(unixDomainSocketPath: sp).wait()
    #if canImport(Darwin)
    chmod(sp, 0o777)
    #elseif canImport(Glibc)
    chmod(sp, 0o777)
    #endif
    print("Listening on Unix socket \(sp)")
} else {
    channel = try bootstrap.bind(host: "0.0.0.0", port: port).wait()
    print("Listening on port \(port)")
}

try channel.closeFuture.wait()
