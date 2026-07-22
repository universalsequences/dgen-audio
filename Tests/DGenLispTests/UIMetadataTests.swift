import XCTest
@testable import DGenLisp
import DGenLazy

final class UIMetadataTests: XCTestCase {
    override func setUp() {
        super.setUp()
        DGenConfig.backend = .c
        DGenConfig.sampleRate = 48_000
        DGenConfig.maxFrameCount = 128
        LazyGraphContext.reset()
    }

    func testManifestIncludesParamUIMetadataGroupsAndEnvelopes() throws {
        let manifest = try compileManifest("""
        (param op1-ratio @group op1 @default 1.0 @min 0.25 @max 16)
        (param op1-attack @group op1 @env op1-env @role attack @default 0.01)
        (param op1-decay @group op1 @env op1-env @role decay @default 0.2)
        (param op1-sustain @group op1 @env op1-env @role sustain @default 0.5)
        (param op1-release @group op1 @env op1-env @role release @default 0.3)
        (param op2-ratio @group op2 @default 2.0 @min 0.25 @max 16)
        (out (+ op1-ratio op2-ratio) 1 @name audio)
        """)

        let paramsByName = Dictionary(uniqueKeysWithValues: manifest.params.map { ($0.name, $0) })
        XCTAssertEqual(paramsByName["op1-ratio"]?.group, "op1")
        XCTAssertNil(paramsByName["op1-ratio"]?.env)
        XCTAssertNil(paramsByName["op1-ratio"]?.role)
        XCTAssertEqual(paramsByName["op1-attack"]?.group, "op1")
        XCTAssertEqual(paramsByName["op1-attack"]?.env, "op1-env")
        XCTAssertEqual(paramsByName["op1-attack"]?.role, "attack")

        XCTAssertEqual(manifest.groups.map(\.name), ["op1", "op2"])
        XCTAssertEqual(manifest.envelopes.count, 1)
        let envelope = try XCTUnwrap(manifest.envelopes.first)
        XCTAssertEqual(envelope.name, "op1-env")
        XCTAssertEqual(envelope.group, "op1")
        XCTAssertEqual(envelope.roles.attack, "op1-attack")
        XCTAssertEqual(envelope.roles.decay, "op1-decay")
        XCTAssertEqual(envelope.roles.sustain, "op1-sustain")
        XCTAssertEqual(envelope.roles.release, "op1-release")
    }

    func testManifestEncodesMissingEnvelopeRolesAsNull() throws {
        let manifest = try compileManifest("""
        (param attack @env amp-env @role attack @default 0.01)
        (param decay @env amp-env @role decay @default 0.2)
        (out (+ attack decay) 1)
        """)

        let json = try JSONEncoder.sortedPrettyString(from: manifest)
        XCTAssertTrue(json.contains(#""envelopes""#))
        XCTAssertTrue(json.contains(#""sustain" : null"#))
        XCTAssertTrue(json.contains(#""release" : null"#))
        XCTAssertTrue(json.contains(#""group" : null"#))
    }

    func testRoleWithoutEnvIsRejected() throws {
        XCTAssertThrowsError(try evaluate("""
        (param attack @role attack @default 0.01)
        """)) { error in
            XCTAssertError(error, contains: "has @role but is missing @env")
        }
    }

    func testEnvWithoutRoleIsRejected() throws {
        XCTAssertThrowsError(try evaluate("""
        (param attack @env amp-env @default 0.01)
        """)) { error in
            XCTAssertError(error, contains: "has @env 'amp-env' but is missing @role")
        }
    }

    func testDuplicateEnvelopeRoleIsRejected() throws {
        XCTAssertThrowsError(try evaluate("""
        (param attack-a @env amp-env @role attack @default 0.01)
        (param attack-b @env amp-env @role attack @default 0.02)
        """)) { error in
            XCTAssertError(error, contains: "duplicate @role attack")
        }
    }

    func testConflictingEnvelopeGroupsAreRejected() throws {
        XCTAssertThrowsError(try evaluate("""
        (param attack @group amp @env amp-env @role attack @default 0.01)
        (param decay @group filter @env amp-env @role decay @default 0.2)
        """)) { error in
            XCTAssertError(error, contains: "conflicting @group values 'amp' and 'filter'")
        }
    }

    func testInvalidRoleIsRejected() throws {
        XCTAssertThrowsError(try evaluate("""
        (param attack @env amp-env @role fast @default 0.01)
        """)) { error in
            XCTAssertError(error, contains: "invalid @role 'fast'")
        }
    }

    private func evaluate(_ source: String) throws {
        let evaluator = LispEvaluator()
        try evaluator.evaluate(nodes: try lowerModulation(in: parseSource(source)))
    }

    private func compileManifest(_ source: String) throws -> PatchManifest {
        let evaluator = LispEvaluator()
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

private extension JSONEncoder {
    static func sortedPrettyString<T: Encodable>(from value: T) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(value)
        return String(decoding: data, as: UTF8.self)
    }
}

