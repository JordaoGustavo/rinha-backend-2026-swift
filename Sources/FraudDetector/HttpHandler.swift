import NIOCore
import NIOHTTP1

public final class HttpHandler: ChannelInboundHandler, @unchecked Sendable {
    public typealias InboundIn = HTTPServerRequestPart
    public typealias OutboundOut = HTTPServerResponsePart

    private let detector: IvfDetector
    private let mlp: MlpDetector?

    private var uri: String = ""
    private var method: HTTPMethod = .GET
    private var bodyBuffer: ByteBuffer?

    // Pre-built JSON body bytes for each outcome:
    // indices 0..5 = approved (fraudCount 0..5)
    // indices 6..11 = denied (fraudCount 0..5)
    private let jsonBodies: [[UInt8]]
    private let readyBody: [UInt8]
    private let notFoundBody: [UInt8]
    private let fallbackBody: [UInt8]

    public init(detector: IvfDetector, mlp: MlpDetector? = nil) {
        self.detector = detector
        self.mlp = mlp

        var bodies = [[UInt8]](repeating: [], count: 12)
        for fraudCount in 0...5 {
            let fraudScore = Float(fraudCount) * 0.2
            let scoreStr = String(format: "%.1f", fraudScore)
            bodies[fraudCount] = Array("{\"approved\":true,\"fraud_score\":\(scoreStr)}".utf8)
            bodies[6 + fraudCount] = Array("{\"approved\":false,\"fraud_score\":\(scoreStr)}".utf8)
        }
        self.jsonBodies = bodies
        self.readyBody = Array("".utf8)
        self.notFoundBody = Array("".utf8)
        // Fallback = approved with 0 fraud
        self.fallbackBody = bodies[0]
    }

    public func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        let part = unwrapInboundIn(data)

        switch part {
        case .head(let head):
            self.method = head.method
            self.uri = head.uri
            self.bodyBuffer = nil

        case .body(let buffer):
            if self.bodyBuffer == nil {
                self.bodyBuffer = buffer
            } else {
                self.bodyBuffer!.writeImmutableBuffer(buffer)
            }

        case .end:
            handleRequest(context: context)
        }
    }

    private func handleRequest(context: ChannelHandlerContext) {
        switch (method, uri) {
        case (.GET, "/ready"):
            sendResponse(context: context, status: .ok, body: readyBody, contentType: nil)

        case (.POST, "/fraud-score"):
            handleFraudScore(context: context)

        default:
            sendResponse(context: context, status: .notFound, body: notFoundBody, contentType: nil)
        }
    }

    private func handleFraudScore(context: ChannelHandlerContext) {
        guard let bodyBuffer = self.bodyBuffer else {
            sendFraudResponse(context: context, approved: true, fraudCount: 0)
            return
        }

        var vector = (
            Float(0), Float(0), Float(0), Float(0),
            Float(0), Float(0), Float(0), Float(0),
            Float(0), Float(0), Float(0), Float(0),
            Float(0), Float(0), Float(0), Float(0)
        )

        bodyBuffer.withUnsafeReadableBytes { bufferPtr in
            let jsonPtr = UnsafeRawBufferPointer(bufferPtr)
            withUnsafeMutablePointer(to: &vector) { vecTuplePtr in
                let vecPtr = UnsafeMutableRawPointer(vecTuplePtr)
                    .assumingMemoryBound(to: Float.self)
                TransactionParser.parse(jsonPtr, into: vecPtr)
            }
        }

        // MLP fast-path: if confident, skip k-NN entirely
        if let mlp = self.mlp {
            let mlpResult = withUnsafePointer(to: &vector) { vecTuplePtr in
                let vecPtr = UnsafeRawPointer(vecTuplePtr)
                    .assumingMemoryBound(to: Float.self)
                return mlp.predict(vector: vecPtr, dims: 14)
            }
            if mlpResult.confident {
                let approved = !mlpResult.isFraud
                let fraudCount = mlpResult.isFraud ? 5 : 0
                sendFraudResponse(context: context, approved: approved, fraudCount: fraudCount)
                return
            }
        }

        // Fallback: full IVF k-NN search
        let result = withUnsafePointer(to: &vector) { vecTuplePtr in
            let vecPtr = UnsafeRawPointer(vecTuplePtr)
                .assumingMemoryBound(to: Float.self)
            return detector.score(vecPtr)
        }

        sendFraudResponse(context: context, approved: result.approved, fraudCount: result.fraudCount)
    }

    @inline(__always)
    private func sendFraudResponse(context: ChannelHandlerContext, approved: Bool, fraudCount: Int) {
        let fc = (fraudCount < 0 || fraudCount > 5) ? 0 : fraudCount
        let body = jsonBodies[approved ? fc : 6 + fc]
        sendResponse(context: context, status: .ok, body: body, contentType: "application/json")
    }

    private func sendResponse(
        context: ChannelHandlerContext,
        status: HTTPResponseStatus,
        body: [UInt8],
        contentType: String?
    ) {
        var headers = HTTPHeaders()
        headers.add(name: "Content-Length", value: "\(body.count)")
        if let ct = contentType {
            headers.add(name: "Content-Type", value: ct)
        }
        let head = HTTPResponseHead(version: .http1_1, status: status, headers: headers)
        context.write(wrapOutboundOut(.head(head)), promise: nil)

        if !body.isEmpty {
            var buf = context.channel.allocator.buffer(capacity: body.count)
            buf.writeBytes(body)
            context.write(wrapOutboundOut(.body(.byteBuffer(buf))), promise: nil)
        }

        context.writeAndFlush(wrapOutboundOut(.end(nil)), promise: nil)
    }

    public func errorCaught(context: ChannelHandlerContext, error: Error) {
        context.close(promise: nil)
    }
}
