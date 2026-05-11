import Foundation
import FraudDetector

struct LoadedData {
    var vectors: [[Float]]
    var labels: [UInt8]
}

enum DataLoader {
    static func load(path: String) throws -> LoadedData {
        guard FileManager.default.fileExists(atPath: path) else {
            throw PreprocessorError.fileNotFound(path)
        }

        let jsonData: Data
        if path.hasSuffix(".gz") {
            print("  Decompressing \(path) via gunzip...")
            let pipe = Pipe()
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/usr/bin/gunzip")
            process.arguments = ["-c", path]
            process.standardOutput = pipe
            try process.run()

            jsonData = pipe.fileHandleForReading.readDataToEndOfFile()
            process.waitUntilExit()
            guard process.terminationStatus == 0 else {
                throw PreprocessorError.decompressionFailed
            }
        } else {
            jsonData = try Data(contentsOf: URL(fileURLWithPath: path))
        }
        print("  Decompressed size: \(jsonData.count / 1_048_576) MB")

        guard let jsonArray = try JSONSerialization.jsonObject(with: jsonData) as? [[String: Any]] else {
            throw PreprocessorError.parseFailed("Root is not a JSON array of objects")
        }

        let count = jsonArray.count
        print("  Parsing \(count) entries...")

        var vectors = [[Float]]()
        vectors.reserveCapacity(3_200_000)
        var labels = [UInt8]()
        labels.reserveCapacity(3_200_000)

        let dimOrder = [6, 10, 9, 5, 11, 2, 4, 7, 0, 1, 8, 12, 3, 13]

        for (i, entry) in jsonArray.enumerated() {
            guard let vectorRaw = entry["vector"] as? [Any] else {
                throw PreprocessorError.parseFailed("Entry \(i) missing 'vector'")
            }
            var orig = [Float](repeating: 0, count: 14)
            for (j, v) in vectorRaw.prefix(14).enumerated() {
                if let d = v as? Double { orig[j] = Float(d) }
                else if let n = v as? NSNumber { orig[j] = n.floatValue }
            }
            var vector = [Float](repeating: 0, count: 14)
            for d in 0..<14 {
                vector[d] = orig[dimOrder[d]]
            }
            vectors.append(vector)

            let labelStr = entry["label"] as? String ?? "legitimate"
            labels.append(labelStr == "fraud" ? 1 : 0)

            if (i + 1) % 500_000 == 0 {
                print("    Parsed \(i + 1)/\(count) entries")
            }
        }

        print("  Done: \(vectors.count) vectors, fraud=\(labels.filter { $0 == 1 }.count)")
        return LoadedData(vectors: vectors, labels: labels)
    }
}
