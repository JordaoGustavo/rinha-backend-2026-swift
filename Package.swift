// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "RinhaBackend",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "RinhaApp", targets: ["RinhaApp"]),
        .executable(name: "Preprocessor", targets: ["Preprocessor"]),
    ],
    dependencies: [],
    targets: [
        .executableTarget(
            name: "RinhaApp",
            dependencies: ["FraudDetector"],
            path: "Sources/RinhaApp"
        ),
        .executableTarget(
            name: "Preprocessor",
            dependencies: ["FraudDetector"],
            path: "Sources/Preprocessor",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .target(
            name: "FraudDetector",
            dependencies: ["CSimd"],
            path: "Sources/FraudDetector"
        ),
        .target(
            name: "CSimd",
            path: "Sources/CSimd",
            cSettings: [
                .unsafeFlags(["-O3"]),
                .unsafeFlags(["-mavx2", "-mfma"], .when(platforms: [.linux])),
            ]
        ),
        .testTarget(
            name: "FraudDetectorTests",
            dependencies: ["FraudDetector"],
            path: "Tests/FraudDetectorTests"
        ),
    ]
)
