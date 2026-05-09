import NIOCore

public final class ResponseCache: @unchecked Sendable {
    private let outcomes: [[UInt8]]
    public let ready: [UInt8]
    public let notFound: [UInt8]
    public let badRequest: [UInt8]

    private init() {
        ready = Self.buildEmpty(status: 200, reason: "OK")
        notFound = Self.buildEmpty(status: 404, reason: "Not Found")
        badRequest = Self.buildEmpty(status: 400, reason: "Bad Request")
        var out = [[UInt8]](repeating: [], count: 12)
        for fraudCount in 0...5 {
            let fraudScore = Float(fraudCount) * 0.2
            let scoreStr = String(format: "%.1f", fraudScore)
            out[fraudCount] = Self.buildFramed(status: 200, reason: "OK", jsonBody: "{\"approved\":true,\"fraud_score\":\(scoreStr)}")
            out[6 + fraudCount] = Self.buildFramed(status: 200, reason: "OK", jsonBody: "{\"approved\":false,\"fraud_score\":\(scoreStr)}")
        }
        outcomes = out
    }

    public static func build() -> ResponseCache { ResponseCache() }

    @inline(__always)
    public func get(approved: Bool, fraudCount: Int) -> [UInt8] {
        let fc = (fraudCount < 0 || fraudCount > 5) ? 0 : fraudCount
        return outcomes[approved ? fc : 6 + fc]
    }

    private static func buildFramed(status: Int, reason: String, jsonBody: String) -> [UInt8] {
        let bodyBytes = Array(jsonBody.utf8)
        let head = "HTTP/1.1 \(status) \(reason)\r\nContent-Length: \(bodyBytes.count)\r\nContent-Type: application/json\r\n\r\n"
        return Array(head.utf8) + bodyBytes
    }

    private static func buildEmpty(status: Int, reason: String) -> [UInt8] {
        return Array("HTTP/1.1 \(status) \(reason)\r\nContent-Length: 0\r\n\r\n".utf8)
    }
}
