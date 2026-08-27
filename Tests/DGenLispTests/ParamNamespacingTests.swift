import XCTest
@testable import DGenLisp
import DGenLazy

final class ParamNamespacingTests: XCTestCase {
    private func message(for error: Error) -> String {
        (error as? LispError)?.message ?? String(describing: error)
    }

    override func setUp() {
        super.setUp()
        DGenConfig.backend = .c
        DGenConfig.sampleRate = 48_000
        DGenConfig.maxFrameCount = 128
        LazyGraphContext.reset()
    }

    func testGroupedParamsResolveByDottedIdentity() throws {
        let evaluator = LispEvaluator()
        try evaluator.evaluate(nodes: parseSource("""
        (param attack @group op1 @default 0.25)
        (param attack @group op2 @default 0.75)
        (out op1.attack 1)
        (out op2.attack 2)
        """))

        XCTAssertEqual(evaluator.outputs[0].signal.memoryCellId, evaluator.params[0].cellId)
        XCTAssertEqual(evaluator.outputs[1].signal.memoryCellId, evaluator.params[1].cellId)
    }

    func testBareReferenceResolvesAUniqueGroupedParam() throws {
        let evaluator = LispEvaluator()
        try evaluator.evaluate(nodes: parseSource("""
        (param attack @group op1 @default 0.25)
        (out attack 1)
        """))

        XCTAssertEqual(evaluator.outputs[0].signal.memoryCellId, evaluator.params[0].cellId)
    }

    func testAmbiguousBareReferenceListsCanonicalCandidates() throws {
        let evaluator = LispEvaluator()
        XCTAssertThrowsError(try evaluator.evaluate(nodes: parseSource("""
        (param attack @group op2 @default 0.75)
        (out attack 1)
        (param attack @group op1 @default 0.25)
        """))) { error in
            XCTAssertTrue(
                self.message(for: error).contains(
                    "ambiguous parameter reference 'attack'; use one of: op1.attack, op2.attack"),
                "unexpected error: \(error)"
            )
        }
    }

    func testDuplicateParamIdentityIsRejected() throws {
        let evaluator = LispEvaluator()
        XCTAssertThrowsError(try evaluator.evaluate(nodes: parseSource("""
        (param attack @group op1 @default 0.25)
        (param attack @group op1 @default 0.75)
        """))) { error in
            XCTAssertTrue(self.message(for: error).contains("duplicate param 'op1.attack'"))
        }
    }

    func testManifestEmitsCanonicalIdShortDisplayNameAndGroup() throws {
        let evaluator = LispEvaluator()
        try evaluator.evaluate(nodes: parseSource("""
        (param attack @group op1 @default 0.25 @min 0 @max 1)
        (out op1.attack 1)
        """))

        let graph = LazyGraphContext.current
        graph.addOutput(evaluator.outputs[0].signal, channel: 0)
        let compilation = try graph.compileOnly(frameCount: 64, voiceCount: 1)
        let manifest = generateManifest(
            compilerResult: CompilerResult(
                dylibPath: "", cSourcePath: "", compilationResult: compilation, cSource: ""),
            evaluator: evaluator,
            options: CompilerOptions(
                outputDir: ".", name: "patch", sampleRate: 48_000,
                maxFrames: 64, voiceCount: 1, debug: false)
        )

        let param = try XCTUnwrap(manifest.params.first)
        XCTAssertEqual(param.name, "op1.attack")
        XCTAssertEqual(param.displayName, "attack")
        XCTAssertEqual(param.group, "op1")
    }

    func testLegacyDottedDeclarationKeepsItsExistingIdentity() throws {
        let nodes = try lowerModulation(in: parseSource("""
        (param fm.attack @group fm @default 0.25 @min 0 @max 1)
        (param attack @group amp @default 0.5 @min 0 @max 1)
        (out fm.attack 1)
        (out attack 2)
        """))
        let evaluator = LispEvaluator()
        try evaluator.evaluate(nodes: nodes)

        XCTAssertEqual(evaluator.params[0].canonicalName, "fm.attack")
        XCTAssertEqual(evaluator.params[0].name, "fm.attack")
        XCTAssertEqual(
            evaluator.outputs[0].signal.memoryCellId,
            evaluator.params[0].cellId
        )
        // A bare `attack` elsewhere stays unambiguous: the legacy declaration
        // keeps its declared name, it does not also claim the short name.
        XCTAssertEqual(evaluator.params[1].canonicalName, "amp.attack")
        XCTAssertEqual(
            evaluator.outputs[1].signal.memoryCellId,
            evaluator.params[1].cellId
        )
    }

    func testTildeSuffixedNonParameterSymbolsSurviveLowering() throws {
        let nodes = try lowerModulation(in: parseSource("""
        (param level @default 0.5 @min 0 @max 1)
        (def scaled~ (* level 2))
        (out scaled~ 1)
        """))
        XCTAssertTrue(
            nodes.contains(.list([.atom("out"), .atom("scaled~"), .atom("1")])),
            "a non-parameter `~` symbol was rewritten: \(nodes)"
        )

        // It still evaluates: `scaled~` is an ordinary `def` binding.
        let evaluator = LispEvaluator()
        try evaluator.evaluate(nodes: nodes)
        XCTAssertEqual(evaluator.outputs.count, 1)
    }

    func testAttributeValuesAreNotRewrittenAsReferences() throws {
        let nodes = try lowerModulation(in: parseSource("""
        (def mod1 (in 5 @name mod1 @modulator 1))
        (param amount @group op1 @default 0.25 @min 0 @max 1 @mod true @mod-mode additive)
        (out (mod op1.amount) 1 @name amount~)
        """))

        let outNode = try XCTUnwrap(nodes.last)
        guard case .list(let elements) = outNode else {
            return XCTFail("expected an out list, got \(outNode)")
        }
        XCTAssertTrue(
            elements.contains(.atom("amount~")),
            "attribute value was rewritten: \(elements)"
        )
    }

    func testDottedModFormsAndTildeSugarLowerIndependently() throws {
        let nodes = try lowerModulation(in: parseSource("""
        (def mod1 (in 5 @name mod1 @modulator 1))
        (param amount @group op1 @default 0.25 @min 0 @max 1 @mod true @mod-mode additive)
        (param amount @group op2 @default 0.75 @min 0 @max 1 @mod true @mod-mode additive)
        (out (mod op1.amount) 1)
        (out op2.amount~ 2)
        """))
        let evaluator = LispEvaluator()
        try evaluator.evaluate(nodes: nodes)

        let canonicalNames = Set(evaluator.params.map(\.canonicalName))
        XCTAssertTrue(canonicalNames.contains("op1.amount"))
        XCTAssertTrue(canonicalNames.contains("op2.amount"))
        XCTAssertTrue(canonicalNames.contains("__mod__op1.amount__active"))
        XCTAssertTrue(canonicalNames.contains("__mod__op2.amount__active"))

        let destinations = evaluator.params.filter { $0.modulationMode != nil }
        XCTAssertEqual(Set(destinations.map(\.canonicalName)), ["op1.amount", "op2.amount"])

        let graph = LazyGraphContext.current
        for output in evaluator.outputs {
            graph.addOutput(output.signal, channel: output.channel)
        }
        let compilation = try graph.compileOnly(frameCount: 64, voiceCount: 1)
        let manifest = generateManifest(
            compilerResult: CompilerResult(
                dylibPath: "", cSourcePath: "", compilationResult: compilation, cSource: ""),
            evaluator: evaluator,
            options: CompilerOptions(
                outputDir: ".", name: "patch", sampleRate: 48_000,
                maxFrames: 64, voiceCount: 1, debug: false)
        )
        XCTAssertEqual(
            Set(manifest.modDestinations.map(\.name)), ["op1.amount", "op2.amount"])
    }
}
