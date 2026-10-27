// swift-tools-version: 6.1
// Stack export: build the server and native decode service from the retained source.
import PackageDescription
let package = Package(
    name: "Mference", platforms: [.macOS(.v15)],
    products: [
        .library(name: "Mference", targets: ["Mference"]),
        .executable(name: "MferenceServer", targets: ["MferenceServer"]),
        .executable(name: "MferenceDecodeService", targets: ["MferenceDecodeService"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        .package(url: "https://github.com/apple/swift-nio.git", exact: "2.99.0"),
    ],
    targets: [
        .target(name: "Mference", dependencies: [.product(name: "Tokenizers", package: "swift-transformers")],
                path: "Sources/Mference", resources: [.copy("Metal")]),
        .target(name: "MferenceDecodeProtocol", path: "Sources/MferenceDecodeProtocol"),
        .target(name: "MferenceAppCore", dependencies: ["Mference", "MferenceDecodeProtocol"],
                path: "Sources/MferenceApp/Core",
                exclude: ["Installation", "State", "Resources", "Configuration/AppPromptPreset.swift", "Configuration/MacAppSettings.swift"],
                sources: ["Configuration/AppContextLengthOption.swift", "Configuration/AppRuntimeOptions.swift", "Diagnostics", "Inference"]),
        .executableTarget(name: "MferenceDecodeService", dependencies: ["MferenceAppCore", "MferenceDecodeProtocol"], path: "Sources/MferenceDecodeService"),
        .target(name: "MferenceServerCore", dependencies: ["Mference", .product(name: "NIOCore", package: "swift-nio"),
                .product(name: "NIOPosix", package: "swift-nio"), .product(name: "NIOHTTP1", package: "swift-nio")], path: "Sources/MferenceServer/Core"),
        .executableTarget(name: "MferenceServer", dependencies: ["MferenceServerCore"], path: "Sources/MferenceServer/Command"),
    ]
)
