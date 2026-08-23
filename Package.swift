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
        .executable(name: "ModalDrum", targets: ["ModalDrum"]),
        .executable(name: "DGenLisp", targets: ["DGenLisp"]),
        .library(name: "DGen", targets: ["DGen"]),
        .library(name: "DGenLazy", targets: ["DGenLazy"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-crypto", from: "3.0.0")
    ],
    targets: [
        .target(
            name: "DGen",
            dependencies: [
                "DGenHostSupport",
                .product(name: "Crypto", package: "swift-crypto", condition: .when(platforms: [.linux, .windows, .android]))
            ],
            linkerSettings: [
                .linkedFramework("Cocoa", .when(platforms: [.macOS])),
                .linkedFramework("Metal", .when(platforms: [.macOS])),
                .linkedFramework("MetalKit", .when(platforms: [.macOS])),
                .linkedFramework("QuartzCore", .when(platforms: [.macOS]))
            ]
        ),
        .target(
            name: "DGenHostSupport",
            path: "Sources/DGenHostSupport",
            publicHeadersPath: "include",
            linkerSettings: [
                .linkedFramework("Accelerate", .when(platforms: [.macOS]))
            ]
        ),
        .executableTarget(
            name: "DDSPE2E",
            dependencies: ["DGenLazy"],
            path: "Examples/DDSPE2E",
            exclude: ["README.md", "scripts"]
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
            dependencies: ["DGenLazy", "DGen", "DGenTrainProtocol"],
            path: "Examples/SynthID",
            exclude: ["SPEC.md", "FDCHECK_FINDING.md", "RUNG1_REMAINING.md", "RUNG2_STATUS.md", "RUNG3_STATUS.md", "RUNG3_BLOCKER.md", "scripts", "targets"]
        ),
        .executableTarget(
            name: "ModalDrum",
            dependencies: ["DGenLazy"],
            path: "Examples/ModalDrum",
            exclude: ["README.md", "scripts"]
        ),
        .executableTarget(
            name: "DGenLisp",
            dependencies: ["DGenLazy", "DGenTrainProtocol"],
            path: "Sources/DGenLisp"
        ),
        .target(
            name: "DGenTrainProtocol",
            path: "Sources/DGenTrainProtocol"
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
            name: "ModalDrumTests",
            dependencies: ["ModalDrum", "DGenLazy"],
            path: "Tests/ModalDrumTests"
        ),
        .testTarget(
            name: "DGenLispTests",
            dependencies: ["DGenLisp", "DGenLazy", "DGenTrainProtocol"],
            path: "Tests/DGenLispTests"
        ),
        .testTarget(
            name: "DGenTrainProtocolTests",
            dependencies: ["DGenTrainProtocol"],
            path: "Tests/DGenTrainProtocolTests",
            resources: [.copy("Fixtures")]
        ),
    ]
)
