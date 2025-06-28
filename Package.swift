// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "DGen",
    platforms: [
        .macOS(.v10_15)
    ],
    products: [
        .executable(name: "dgen", targets: ["DGenApp"]),
        .library(name: "DGen", targets: ["DGen"]),
    ],
    targets: [
        .target(
            name: "DGen",
            linkerSettings: [
                .linkedFramework("Cocoa"),
                .linkedFramework("Metal"),
                .linkedFramework("MetalKit"),
                .linkedFramework("QuartzCore")
            ]
        ),
        .executableTarget(
            name: "DGenApp",
            dependencies: ["DGen"],
            path: "Sources/DGenApp"
        ),
        .testTarget(
            name: "DGenTests",
            dependencies: ["DGen"],
            path: "Tests/DGenTests"
        ),
    ]
)
