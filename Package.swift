// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "DGen",
    platforms: [
        .macOS(.v10_15)
    ],
    products: [
        .executable(name: "DDSPE2E", targets: ["DDSPE2E"]),
        .library(name: "DGen", targets: ["DGen"]),
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
        .executableTarget(
            name: "DDSPE2E",
            dependencies: ["DGenLazy"],
            path: "Examples/DDSPE2E",
            exclude: ["README.md"]
        ),
        .target(
            name: "DGenLazy",
            dependencies: ["DGen"]
        ),
        .testTarget(
            name: "DGenTests",
            dependencies: ["DGen"],
            path: "Tests/DGenTests"
        ),
        .testTarget(
            name: "DGenLazyTests",
            dependencies: ["DGenLazy", "DGen", "DDSPE2E"],
            path: "Tests/DGenLazyTests"
        ),
    ]
)
