import XCTest
@testable import DGenLisp
import DGenLazy

/// `@options` declares a param a labeled discrete choice: inline labels are
/// baked into the manifest, a tensor binding stays a reference the host
/// re-resolves against the asset backing it.
final class ParamOptionsTests: XCTestCase {
    private var tempDir: URL!

    override func setUpWithError() throws {
        try super.setUpWithError()
        DGenConfig.backend = .c
        DGenConfig.sampleRate = 48_000
        DGenConfig.maxFrameCount = 64
        LazyGraphContext.reset()
        tempDir = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("dgenlisp-options-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let tempDir {
            try? FileManager.default.removeItem(at: tempDir)
        }
        try super.tearDownWithError()
    }

    func testInlineLabelsAreBakedAndDeriveTheDomain() throws {
        let manifest = try compileManifest("""
        (param mode @options ["lowpass" "highpass" "bandpass" "notch"] @default 0)
        (out mode 1 @name audio)
        """)

        let param = try XCTUnwrap(manifest.params.first { $0.name == "mode" })
        XCTAssertEqual(param.options?.labels, ["lowpass", "highpass", "bandpass", "notch"])
        XCTAssertNil(param.options?.tensor)
        XCTAssertEqual(param.min, 0)
        XCTAssertEqual(param.max, 3)
    }

    func testInlineLabelsAcceptBareTokens() throws {
        let manifest = try compileManifest("""
        (param mode @options [lowpass highpass])
        (out mode 1 @name audio)
        """)

        XCTAssertEqual(manifest.params.first?.options?.labels, ["lowpass", "highpass"])
    }

    func testAuthoredRangeIsIgnoredForOptionParams() throws {
        let manifest = try compileManifest("""
        (param mode @options ["a" "b"] @min 3 @max 99)
        (out mode 1 @name audio)
        """)

        let param = try XCTUnwrap(manifest.params.first)
        XCTAssertEqual(param.min, 0)
        XCTAssertEqual(param.max, 1)
    }

    func testTensorReferenceEmitsAssetVariantAndDerivesDomainFromSets() throws {
        try writeBank(sets: ["Basic Shapes", "Harmonics"], wavesPerSet: 2)

        let manifest = try compileManifest("""
        (def bank (tensor @shape [2 4] @file "waves/bank.json"))
        (param table @options bank @default 0)
        (out (+ table (peek bank 0 0)) 1 @name audio)
        """, sourceDirectory: tempDir)

        let param = try XCTUnwrap(manifest.params.first { $0.name == "table" })
        let options = try XCTUnwrap(param.options)
        XCTAssertNil(options.labels)
        XCTAssertEqual(options.tensor, "bank")
        XCTAssertEqual(options.file, "waves/bank.json")
        XCTAssertEqual(options.key, "sets")
        XCTAssertEqual(param.min, 0)
        XCTAssertEqual(param.max, 1)
    }

    func testOptionsKeySelectsAnotherMetadataList() throws {
        try writeBank(sets: ["Only"], wavesPerSet: 4, waveNames: ["a", "b", "c", "d"])

        let manifest = try compileManifest("""
        (def bank (tensor @shape [2 4] @file "waves/bank.json"))
        (param table @options bank @options-key wave-names)
        (out (+ table (peek bank 0 0)) 1 @name audio)
        """, sourceDirectory: tempDir)

        let options = try XCTUnwrap(manifest.params.first { $0.name == "table" }?.options)
        XCTAssertEqual(options.key, "wave_names")
        XCTAssertEqual(manifest.params.first { $0.name == "table" }?.max, 3)
    }

    func testMissingLabelListsFallBackToNumberedSets() throws {
        try writeBank(sets: nil, wavesPerSet: 2)

        let manifest = try compileManifest("""
        (def bank (tensor @shape [2 4] @file "waves/bank.json"))
        (param table @options bank)
        (out (+ table (peek bank 0 0)) 1 @name audio)
        """, sourceDirectory: tempDir)

        // 4 lanes / waves_per_set 2 = 2 numbered sets.
        XCTAssertEqual(manifest.params.first { $0.name == "table" }?.max, 1)
    }

    func testTensorWithoutFileContentsIsACompileError() throws {
        let evaluator = LispEvaluator(sourceDirectory: tempDir)
        XCTAssertThrowsError(try evaluator.evaluate(nodes: parseSource("""
        (def bank (tensor @shape [2 4] @data [0 0 0 0 0 0 0 0]))
        (param table @options bank)
        """))) { error in
            XCTAssertError(error, contains: "has no file contents")
        }
    }

    func testUnknownTensorNameIsACompileError() throws {
        let evaluator = LispEvaluator(sourceDirectory: tempDir)
        XCTAssertThrowsError(try evaluator.evaluate(nodes: parseSource("""
        (param table @options nope)
        """))) { error in
            XCTAssertError(error, contains: "does not name a tensor binding")
        }
    }

    func testUnknownAttributesArePassedThroughAsInertHints() throws {
        let manifest = try compileManifest("""
        (param cutoff @default 1000 @min 20 @max 20000 @curve exp @tooltip "filter cutoff")
        (out cutoff 1 @name audio)
        """)

        let attrs = try XCTUnwrap(manifest.params.first?.attrs)
        XCTAssertEqual(attrs["curve"], "exp")
        XCTAssertEqual(attrs["tooltip"], "filter cutoff")
        XCTAssertNil(attrs["default"])
        XCTAssertNil(attrs["min"])
    }

    func testParamsWithoutExtraAttributesOmitTheMap() throws {
        let manifest = try compileManifest("""
        (param gain @default 0.5 @min 0 @max 1)
        (out gain 1 @name audio)
        """)

        XCTAssertNil(manifest.params.first?.attrs)
        XCTAssertNil(manifest.params.first?.options)
    }

    // MARK: - Helpers

    private func writeBank(
        sets: [String]?, wavesPerSet: Int, waveNames: [String]? = nil
    ) throws {
        let wavesDir = tempDir.appendingPathComponent("waves", isDirectory: true)
        try FileManager.default.createDirectory(at: wavesDir, withIntermediateDirectories: true)
        var object: [String: Any] = [
            "shape": [2, 4],
            "data": [0.0, 0.25, 0.5, 0.75, 1.0, 0.5, 0.0, -0.5],
            "kind": "wavetable-bank",
            "waves_per_set": wavesPerSet,
        ]
        if let sets { object["sets"] = sets }
        if let waveNames { object["wave_names"] = waveNames }
        let data = try JSONSerialization.data(withJSONObject: object)
        try data.write(to: wavesDir.appendingPathComponent("bank.json"))
    }

    private func compileManifest(
        _ source: String, sourceDirectory: URL? = nil
    ) throws -> PatchManifest {
        let evaluator = LispEvaluator(
            sourceDirectory: sourceDirectory
                ?? URL(fileURLWithPath: FileManager.default.currentDirectoryPath))
        try evaluator.evaluate(nodes: try lowerModulation(in: parseSource(source)))

        let graph = LazyGraphContext.current
        for output in evaluator.outputs {
            graph.addOutput(output.signal, channel: output.channel)
        }
        let compilation = try graph.compileOnly(frameCount: 64, voiceCount: 1)
        return generateManifest(
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
    }

    private func XCTAssertError(
        _ error: Error,
        contains expected: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        guard let lispError = error as? LispError else {
            return XCTFail("unexpected error: \(error)", file: file, line: line)
        }
        XCTAssertTrue(
            lispError.message.contains(expected),
            "expected error containing '\(expected)', got '\(lispError.message)'",
            file: file,
            line: line
        )
    }
}
