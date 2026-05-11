#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#elseif canImport(Musl)
import Musl
#endif

import Foundation

public final class RawServer: @unchecked Sendable {
    private let detector: IvfDetector
    private let mlp: MlpDetector?
    private let responses: ResponseCache
    private let socketPath: String?
    private let port: Int

    // Maps current (variance-reordered) positions back to old (MLP training) order
    private static let reverseDimOrder = [8, 9, 5, 12, 6, 3, 0, 7, 10, 2, 1, 4, 11, 13]

    public init(detector: IvfDetector, mlp: MlpDetector? = nil, socketPath: String? = nil, port: Int = 8080) {
        self.detector = detector
        self.mlp = mlp
        self.responses = ResponseCache.build()
        self.socketPath = socketPath
        self.port = port
    }

    public func run() {
        let fd: Int32
        if let sp = socketPath {
            fd = socket(AF_UNIX, Int32(1), 0)
            guard fd >= 0 else { fatalError("socket() failed: \(errno)") }

            unlink(sp)
            var addr = sockaddr_un()
            addr.sun_family = sa_family_t(AF_UNIX)
            let pathBound = MemoryLayout.size(ofValue: addr.sun_path)
            _ = withUnsafeMutablePointer(to: &addr.sun_path) { ptr in
                sp.withCString { src in
                    memcpy(ptr, src, min(sp.utf8.count, pathBound - 1))
                }
            }
            let bindResult = withUnsafePointer(to: &addr) { addrPtr in
                addrPtr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sPtr in
                    bind(fd, sPtr, socklen_t(MemoryLayout<sockaddr_un>.size))
                }
            }
            guard bindResult == 0 else { fatalError("bind(\(sp)) failed: \(errno)") }
            chmod(sp, 0o777)
            print("Listening on Unix socket \(sp)")
        } else {
            fd = socket(AF_INET, Int32(1), 0)
            guard fd >= 0 else { fatalError("socket() failed: \(errno)") }
            var opt: Int32 = 1
            setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, socklen_t(MemoryLayout<Int32>.size))
            var addr = sockaddr_in()
            addr.sin_family = sa_family_t(AF_INET)
            addr.sin_port = UInt16(port).bigEndian
            addr.sin_addr = in_addr(s_addr: INADDR_ANY.bigEndian)
            let bindResult = withUnsafePointer(to: &addr) { addrPtr in
                addrPtr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sPtr in
                    bind(fd, sPtr, socklen_t(MemoryLayout<sockaddr_in>.size))
                }
            }
            guard bindResult == 0 else { fatalError("bind(:\(port)) failed: \(errno)") }
            print("Listening on port \(port)")
        }

        guard listen(fd, 2048) == 0 else { fatalError("listen() failed: \(errno)") }

        let numWorkers = 16
        for _ in 0..<(numWorkers - 1) {
            let t = Thread { [self] in self.acceptLoop(fd) }
            t.stackSize = 256 * 1024
            t.start()
        }
        acceptLoop(fd)
    }

    private func acceptLoop(_ listenFd: Int32) {
        while true {
            var clientAddr = sockaddr_storage()
            var clientLen = socklen_t(MemoryLayout<sockaddr_storage>.size)
            let clientFd = withUnsafeMutablePointer(to: &clientAddr) { addrPtr in
                addrPtr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sPtr in
                    accept(listenFd, sPtr, &clientLen)
                }
            }
            guard clientFd >= 0 else { continue }
            handleConnection(clientFd)
        }
    }

    private func handleConnection(_ fd: Int32) {
        let bufSize = 8192
        let buf = UnsafeMutablePointer<UInt8>.allocate(capacity: bufSize)
        defer {
            buf.deallocate()
            close(fd)
        }

        var filled = 0
        while true {
            let n = recv(fd, buf + filled, bufSize - filled, 0)
            if n <= 0 { return }
            filled += n

            while filled > 0 {
                let consumed = processOne(fd, buf, filled)
                if consumed < 0 { return }
                if consumed == 0 { break }
                let rem = filled - consumed
                if rem > 0 { memmove(buf, buf + consumed, rem) }
                filled = rem
            }

            if filled == bufSize { return }
        }
    }

    @inline(__always)
    private func processOne(_ fd: Int32, _ buf: UnsafeMutablePointer<UInt8>, _ length: Int) -> Int {
        let headersEnd = memmem4(buf, length)
        if headersEnd < 0 { return 0 }
        let bodyStart = headersEnd + 4

        var rlEnd = 0
        while rlEnd < length && buf[rlEnd] != 0x0A { rlEnd += 1 }
        if rlEnd < 2 || buf[rlEnd - 1] != 0x0D { return -1 }

        let sp1 = firstSpace(buf, rlEnd)
        if sp1 < 0 { return -1 }

        if sp1 == 3 && buf[0] == 0x47 && buf[1] == 0x45 && buf[2] == 0x54 {
            if matchPath(buf + sp1 + 1, rlEnd - sp1 - 1, "/ready") {
                sendAll(fd, responses.ready)
            } else {
                sendAll(fd, responses.notFound)
            }
            return bodyStart
        }

        if sp1 == 4 && buf[0] == 0x50 && buf[1] == 0x4F && buf[2] == 0x53 && buf[3] == 0x54 {
            if matchPath(buf + sp1 + 1, rlEnd - sp1 - 1, "/fraud-score") {
                let cl = parseContentLength(buf, headersEnd)
                if cl < 0 || cl > 4096 {
                    sendAll(fd, responses.badRequest)
                    return -1
                }
                if length < bodyStart + cl { return 0 }
                handleFraudScore(fd, buf + bodyStart, cl)
                return bodyStart + cl
            }
            sendAll(fd, responses.notFound)
            return bodyStart
        }

        sendAll(fd, responses.notFound)
        return bodyStart
    }

    @inline(__always)
    private func handleFraudScore(_ fd: Int32, _ body: UnsafeMutablePointer<UInt8>, _ bodyLen: Int) {
        var vector = (
            Float(0), Float(0), Float(0), Float(0),
            Float(0), Float(0), Float(0), Float(0),
            Float(0), Float(0), Float(0), Float(0),
            Float(0), Float(0), Float(0), Float(0)
        )

        let jsonPtr = UnsafeRawBufferPointer(start: body, count: bodyLen)
        withUnsafeMutablePointer(to: &vector) { vPtr in
            let fp = UnsafeMutableRawPointer(vPtr).assumingMemoryBound(to: Float.self)
            TransactionParser.parse(jsonPtr, into: fp)
        }

        if let mlp = self.mlp {
            var mlpVec = (
                Float(0), Float(0), Float(0), Float(0), Float(0), Float(0), Float(0),
                Float(0), Float(0), Float(0), Float(0), Float(0), Float(0), Float(0)
            )
            withUnsafePointer(to: &vector) { srcPtr in
                let src = UnsafeRawPointer(srcPtr).assumingMemoryBound(to: Float.self)
                withUnsafeMutablePointer(to: &mlpVec) { dstPtr in
                    let dst = UnsafeMutableRawPointer(dstPtr).assumingMemoryBound(to: Float.self)
                    for i in 0..<14 {
                        dst[i] = src[Self.reverseDimOrder[i]]
                    }
                }
            }
            let mlpResult = withUnsafePointer(to: &mlpVec) { ptr in
                let fp = UnsafeRawPointer(ptr).assumingMemoryBound(to: Float.self)
                return mlp.predict(vector: fp, dims: 14)
            }
            if mlpResult.confident {
                let approved = !mlpResult.isFraud
                let fc = mlpResult.isFraud ? 5 : 0
                sendAll(fd, responses.get(approved: approved, fraudCount: fc))
                return
            }
        }

        let result = withUnsafePointer(to: &vector) { vPtr in
            let fp = UnsafeRawPointer(vPtr).assumingMemoryBound(to: Float.self)
            return detector.score(fp)
        }

        let fc = min(max(result.fraudCount, 0), 5)
        sendAll(fd, responses.get(approved: result.approved, fraudCount: fc))
    }

    @inline(__always)
    private func memmem4(_ buf: UnsafePointer<UInt8>, _ len: Int) -> Int {
        guard len >= 4 else { return -1 }
        for i in 0...(len - 4) {
            if buf[i] == 0x0D && buf[i+1] == 0x0A && buf[i+2] == 0x0D && buf[i+3] == 0x0A {
                return i
            }
        }
        return -1
    }

    @inline(__always)
    private func firstSpace(_ buf: UnsafePointer<UInt8>, _ len: Int) -> Int {
        for i in 0..<len {
            if buf[i] == 0x20 { return i }
        }
        return -1
    }

    @inline(__always)
    private func matchPath(_ buf: UnsafePointer<UInt8>, _ len: Int, _ path: StaticString) -> Bool {
        let pathLen = path.utf8CodeUnitCount
        guard len > pathLen else { return false }
        if buf[pathLen] != 0x20 { return false }
        return path.withUTF8Buffer { expected in
            memcmp(buf, expected.baseAddress!, pathLen) == 0
        }
    }

    @inline(__always)
    private func parseContentLength(_ buf: UnsafePointer<UInt8>, _ headersEnd: Int) -> Int {
        let cl16 = Array("Content-Length:".utf8)
        let cl16Lower = Array("content-length:".utf8)
        let needle = cl16.count

        for i in 0..<(headersEnd - needle) {
            var match = true
            for j in 0..<needle {
                if buf[i + j] != cl16[j] && buf[i + j] != cl16Lower[j] {
                    match = false
                    break
                }
            }
            if !match { continue }

            var p = i + needle
            while p < headersEnd && buf[p] == 0x20 { p += 1 }
            var v = 0
            while p < headersEnd {
                let d = Int(buf[p]) - 48
                guard d >= 0 && d <= 9 else { break }
                v = v * 10 + d
                if v > 65536 { return -1 }
                p += 1
            }
            return v
        }
        return -1
    }

    @inline(__always)
    private func sendAll(_ fd: Int32, _ data: [UInt8]) {
        data.withUnsafeBufferPointer { bufPtr in
            guard let base = bufPtr.baseAddress else { return }
            var sent = 0
            let total = data.count
            while sent < total {
                let n = send(fd, base + sent, total - sent, 0)
                if n <= 0 { return }
                sent += n
            }
        }
    }
}
