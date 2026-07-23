import XCTest
@testable import DGenLisp
import DGenLazy

final class WavetableTensorTests: XCTestCase {
    private var tempDir: URL!

    override func setUpWithError() throws {
        try super.setUpWithError()
        DGenConfig.backend = .c
        DGenConfig.sampleRate = 48_000
        DGenConfig.maxFrameCount = 64
        LazyGraphContext.reset()
        tempDir = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("dgenlisp-wavetable-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let tempDir {
            try? FileManager.default.removeItem(at: tempDir)
        }
        try super.tearDownWithError()
    }

    func testWavetableLoadsRelativeJsonAndExportsTensorMetadata() throws {
        let wavesDir = tempDir.appendingPathComponent("waves", isDirectory: true)
        try FileManager.default.createDirectory(at: wavesDir, withIntermediateDirectories: true)
        let tableFile = wavesDir.appendingPathComponent("tiny.json")
        try """
        {
          "shape": [2, 4],
          "data": [
            [0.0, 0.25, 0.5, 0.75],
            [1.0, 0.5, 0.0, -0.5]
          ]
        }
        """.write(to: tableFile, atomically: true, encoding: .utf8)

        let evaluator = LispEvaluator(sourceDirectory: tempDir)
        try evaluator.evaluate(nodes: parseSource("""
        (def waves (wavetable @shape [2 4] @file "waves/tiny.json"))
        (out (peek waves 1 0) 1 @name audio)
        """))

        let graph = LazyGraphContext.current
        for output in evaluator.outputs {
            graph.addOutput(output.signal, channel: output.channel)
        }
        let compilation = try graph.compileOnly(frameCount: 8, voiceCount: 1)
        let manifest = generateManifest(
            compilerResult: CompilerResult(
                dylibPath: "",
                cSourcePath: "",
                compilationResult: compilation,
                cSource: ""
            ),
            evaluator: evaluator,
            options: CompilerOptions(
                outputDir: ".",
                name: "patch",
                sampleRate: 48_000,
                maxFrames: 64,
                voiceCount: 1,
                debug: false
            )
        )

        XCTAssertEqual(manifest.version, 3)
        XCTAssertEqual(manifest.processAbi, "dgen-host-abi-v1")
        XCTAssertEqual(manifest.tensors.count, 1)
        XCTAssertEqual(manifest.tensors[0].name, "waves")
        XCTAssertEqual(manifest.tensors[0].shape, [2, 4])
        XCTAssertEqual(manifest.tensors[0].kind, "wavetable")
        XCTAssertEqual(manifest.tensors[0].mutable, false)
        XCTAssertEqual(manifest.tensors[0].sourceFile, "waves/tiny.json")
        XCTAssertNil(manifest.tensors[0].sourceSampleRate)
        XCTAssertEqual(manifest.tensorInitData.count, 1)
        XCTAssertEqual(manifest.tensorInitData[0].data, [0.0, 0.25, 0.5, 0.75, 1.0, 0.5, 0.0, -0.5])
    }

    func testWavetableRejectsWrongShape() throws {
        let tableFile = tempDir.appendingPathComponent("bad.json")
        try "[1, 2, 3]".write(to: tableFile, atomically: true, encoding: .utf8)

        let evaluator = LispEvaluator(sourceDirectory: tempDir)
        XCTAssertThrowsError(try evaluator.evaluate(nodes: parseSource("""
        (def waves (wavetable @shape [2 4] @file "bad.json"))
        (out (peek waves 0 0) 1)
        """))) { error in
            guard let lispError = error as? LispError else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertTrue(lispError.message.contains("expected 8"))
        }
    }

    func testWavetableModulatedParamBroadcastsScalarCellsInSIMDLoop() throws {
        let wavesDir = tempDir.appendingPathComponent("waves", isDirectory: true)
        try FileManager.default.createDirectory(at: wavesDir, withIntermediateDirectories: true)
        let tableFile = wavesDir.appendingPathComponent("tiny.json")
        try """
        {
          "shape": [4, 1],
          "data": [
            [0.0],
            [0.25],
            [0.5],
            [0.75]
          ]
        }
        """.write(to: tableFile, atomically: true, encoding: .utf8)

        let evaluator = LispEvaluator(sourceDirectory: tempDir)
        let nodes = try lowerModulation(in: parseSource("""
        (def mod1 (in 5 @name mod1 @modulator 1))
        (def waves (wavetable @shape [4 1] @file "waves/tiny.json"))
        (param cutoff @default 7400 @min 80 @max 12000 @unit Hz @mod true @mod-mode additive)
        (def phase (phasor 440))
        (def wave (peek waves (* (wrap phase 0 1) 4) 0))
        (def shaped (* wave wave))
        (out (+ shaped (* (mod cutoff) 0.000001)) 1 @name audio)
        """))
        try evaluator.evaluate(nodes: nodes)

        let graph = LazyGraphContext.current
        for output in evaluator.outputs {
            graph.addOutput(output.signal, channel: output.channel)
        }

        let compilation = try graph.compileOnly(frameCount: 64, voiceCount: 1)
        let source = compilation.kernels.map(\.source).joined(separator: "\n")
        let cutoffCell = try XCTUnwrap(evaluator.params.first { $0.name == "cutoff" }?.cellId)
        let cutoffPhysicalCell = compilation.cellAllocations.cellMappings[cutoffCell] ?? cutoffCell
        let cutoffRead = "memory[\(cutoffPhysicalCell)"
        let cutoffReadRange = try XCTUnwrap(source.range(of: cutoffRead))
        let excerptStart = source.index(cutoffReadRange.lowerBound, offsetBy: -120, limitedBy: source.startIndex) ?? source.startIndex
        let excerptEnd = source.index(cutoffReadRange.upperBound, offsetBy: 240, limitedBy: source.endIndex) ?? source.endIndex
        let excerpt = source[excerptStart..<excerptEnd]

        XCTAssertTrue(
            source.contains("vdupq_n_f32(\(cutoffRead) +"),
            "span-1 params used in SIMD must be scalar-loaded and broadcast; cutoffRead=\(cutoffRead), excerpt=\(excerpt)")
        XCTAssertFalse(
            source.contains("vld1q_f32(&\(cutoffRead))"),
            "span-1 params must not be loaded as contiguous SIMD cells; cutoffRead=\(cutoffRead), excerpt=\(excerpt)")
    }
}
