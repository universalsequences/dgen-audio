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
        .library(name: "DGenFrontend", targets: ["DGenFrontend"]),
        .library(name: "DGenLazy", targets: ["DGenLazy"]),
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
        .target(
            name: "DGenFrontend",
            dependencies: ["DGen"]
        ),
        .executableTarget(
            name: "DGenApp",
            dependencies: ["DGen"],
            path: "Sources/DGenApp"
        ),
        .target(
            name: "DGenLazy",
            dependencies: ["DGen"]
        ),
        .testTarget(
            name: "DGenTests",
            dependencies: ["DGen", "DGenFrontend"],
            path: "Tests/DGenTests"
        ),
        .testTarget(
            name: "DGenLazyTests",
            dependencies: ["DGenLazy", "DGen"],
            path: "Tests/DGenLazyTests"
        ),
    ]
)
