// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "DGen",
    platforms: [
        .macOS(.v10_15)
    ],
    products: [
        .executable(name: "DDSPE2E", targets: ["DDSPE2E"]),
        .executable(name: "HarmonicE2E", targets: ["HarmonicE2E"]),
        .executable(name: "BendingMetal", targets: ["BendingMetal"]),
        .executable(name: "TrainKick808", targets: ["TrainKick808"]),
        .executable(name: "SynthID", targets: ["SynthID"]),
        .executable(name: "DGenLisp", targets: ["DGenLisp"]),
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
        .executableTarget(
            name: "HarmonicE2E",
            dependencies: ["DGenLazy"],
            path: "Examples/HarmonicE2E",
            exclude: ["README.md"]
        ),
        .executableTarget(
            name: "BendingMetal",
            dependencies: ["DGenLazy"],
            path: "Examples/BendingMetal"
        ),
        .executableTarget(
            name: "TrainKick808",
            dependencies: ["DGenLazy", "DGen"],
            path: "Examples/TrainKick808",
            exclude: ["renders", "waveform_compare.py", "EXPERIMENTS.md"]
        ),
        .executableTarget(
            name: "SynthID",
            dependencies: ["DGenLazy", "DGen"],
            path: "Examples/SynthID",
            exclude: ["SPEC.md", "FDCHECK_FINDING.md", "RUNG1_REMAINING.md", "RUNG2_STATUS.md", "RUNG3_STATUS.md", "RUNG3_BLOCKER.md", "scripts", "targets"]
        ),
        .executableTarget(
            name: "DGenLisp",
            dependencies: ["DGenLazy"],
            path: "Sources/DGenLisp"
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
            dependencies: ["DGenLazy", "DGen", "DDSPE2E", "HarmonicE2E"],
            path: "Tests/DGenLazyTests"
        ),
        .testTarget(
            name: "DGenLispTests",
            dependencies: ["DGenLisp", "DGenLazy"],
            path: "Tests/DGenLispTests"
        ),
    ]
)
