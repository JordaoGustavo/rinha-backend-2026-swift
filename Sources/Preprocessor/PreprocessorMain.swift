import Foundation
import FraudDetector

@main
struct PreprocessorApp {
    static func main() throws {
        let args = CommandLine.arguments

        guard args.count >= 2 else {
            printUsage()
            throw PreprocessorError.invalidArguments
        }

        if args[1] == "gen-warmup" {
            guard args.count >= 3 else {
                print("Usage: Preprocessor gen-warmup <output.ndjson> [count=200]")
                throw PreprocessorError.invalidArguments
            }
            let outputPath = args[2]
            let count = parseIntArg(args, prefix: "count=", default: 200)
            try GenWarmup.generate(outputPath: outputPath, count: count)
            return
        }

        guard args.count >= 3 else {
            printUsage()
            throw PreprocessorError.invalidArguments
        }

        let inputPath = args[1]
        let outputPath = args[2]
        let clusters = parseIntArg(args, prefix: "clusters=", default: 4096)
        let kmeansIter = parseIntArg(args, prefix: "kmeans_iter=", default: 20)
        let format = parseStringArg(args, prefix: "format=", default: "ivf")
        let nprobe = parseIntArg(args, prefix: "nprobe=", default: 40)

        print("Loading data from \(inputPath)...")
        let t0 = CFAbsoluteTimeGetCurrent()
        let data = try DataLoader.load(path: inputPath)
        let t1 = CFAbsoluteTimeGetCurrent()
        print("Loaded \(data.vectors.count) vectors in \(String(format: "%.2f", t1 - t0))s")

        switch format {
        case "exact":
            try buildExact(data: data, outputPath: outputPath)
        case "ivf":
            try buildIvf(data: data, outputPath: outputPath, clusters: clusters,
                         kmeansIter: kmeansIter, nprobe: nprobe)
        default:
            print("Unknown format: \(format). Use 'ivf' or 'exact'.")
            throw PreprocessorError.invalidArguments
        }
    }

    static func buildExact(data: LoadedData, outputPath: String) throws {
        let pd = ExactBinaryFormat.paddedDims
        let n = data.vectors.count

        print("Building exact index with \(n) vectors, padding to \(pd) dims...")

        var padded = [Float](repeating: 0, count: n * pd)
        let dims = IvfBinaryFormat.dims
        for i in 0..<n {
            let src = data.vectors[i]
            for d in 0..<min(dims, src.count) {
                padded[i * pd + d] = src[d]
            }
        }

        try ExactBinaryWriter.write(
            vectors: padded,
            labels: data.labels,
            numVectors: n,
            to: outputPath
        )

        print("Wrote exact index to \(outputPath)")
    }

    static func buildIvf(data: LoadedData, outputPath: String, clusters: Int,
                         kmeansIter: Int, nprobe: Int) throws {
        print("Building IVF index: \(data.vectors.count) vectors, \(clusters) clusters, \(kmeansIter) iterations...")

        let t0 = CFAbsoluteTimeGetCurrent()
        let result = IvfBuilder.build(
            vectors: data.vectors,
            labels: data.labels,
            numClusters: clusters,
            kmeansIterations: kmeansIter
        )
        let t1 = CFAbsoluteTimeGetCurrent()
        print("IVF build completed in \(String(format: "%.2f", t1 - t0))s")

        let nprobeFast = min(nprobe, clusters)
        let nprobeFull = min(nprobe * 2, clusters)

        try IvfBinaryWriter.write(result: result, nprobeFast: nprobeFast,
                                  nprobeFull: nprobeFull, to: outputPath)
        print("Wrote IVF index to \(outputPath)")
    }

    static func printUsage() {
        print("Usage:")
        print("  Preprocessor <references.json.gz> <output.bin> [clusters=4096] [kmeans_iter=20] [format=ivf] [nprobe=40]")
        print("  Preprocessor gen-warmup <output.ndjson> [count=200]")
    }

    static func parseIntArg(_ args: [String], prefix: String, default defaultValue: Int) -> Int {
        for arg in args {
            if arg.hasPrefix(prefix) {
                let value = String(arg.dropFirst(prefix.count))
                if let v = Int(value) { return v }
            }
        }
        return defaultValue
    }

    static func parseStringArg(_ args: [String], prefix: String, default defaultValue: String) -> String {
        for arg in args {
            if arg.hasPrefix(prefix) {
                return String(arg.dropFirst(prefix.count))
            }
        }
        return defaultValue
    }
}

enum PreprocessorError: Error {
    case invalidArguments
    case fileNotFound(String)
    case decompressionFailed
    case parseFailed(String)
    case writeFailed(String)
}
