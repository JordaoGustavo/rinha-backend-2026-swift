// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "RinhaBackend",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(url: "https://github.com/apple/swift-nio.git", from: "2.80.0"),
    ],
    targets: [
        .executableTarget(
            name: "RinhaApp",
            dependencies: [
                "FraudDetector",
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOPosix", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio"),
            ],
            path: "Sources/RinhaApp"
        ),
        .executableTarget(
            name: "Preprocessor",
            dependencies: ["FraudDetector"],
            path: "Sources/Preprocessor"
        ),
        .target(
            name: "FraudDetector",
            dependencies: [
                "CSimd",
                .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOPosix", package: "swift-nio"),
                .product(name: "NIOHTTP1", package: "swift-nio"),
            ],
            path: "Sources/FraudDetector"
        ),
        .target(
            name: "CSimd",
            path: "Sources/CSimd",
            cSettings: [.unsafeFlags(["-O3"])]
        ),
        .testTarget(
            name: "FraudDetectorTests",
            dependencies: ["FraudDetector"],
            path: "Tests/FraudDetectorTests"
        ),
    ]
)
