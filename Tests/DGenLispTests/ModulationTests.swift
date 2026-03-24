import XCTest
@testable import DGenLisp
import DGenLazy

final class ModulationTests: XCTestCase {
    private func flattenAtoms(_ nodes: [ASTNode]) -> [String] {
        var atoms: [String] = []

        func walk(_ node: ASTNode) {
            switch node {
            case .atom(let value):
                atoms.append(value)
            case .list(let elements):
                elements.forEach(walk)
            }
        }

        nodes.forEach(walk)
        return atoms
    }

    override func setUp() {
        super.setUp()
        DGenConfig.backend = .c
        DGenConfig.sampleRate = 48_000
        DGenConfig.maxFrameCount = 128
        LazyGraphContext.reset()
    }

    func testLoweringGeneratesHiddenParamsAndResolvedSymbol() throws {
        let source = """
        (def mod1 (in 5 @name mod1 @modulator 1))
        (param cutoff @default 2400 @min 60 @max 12000 @unit Hz @mod true @mod-mode additive)
        (out (mod cutoff) 1)
        """

        let lowered = try lowerModulation(in: parseSource(source))
        let loweredAtoms = flattenAtoms(lowered)

        XCTAssertTrue(loweredAtoms.contains("__mod__cutoff__source"))
        XCTAssertTrue(loweredAtoms.contains("__mod__cutoff__depth"))
        XCTAssertTrue(loweredAtoms.contains("__mod__cutoff__resolved"))
        XCTAssertTrue(loweredAtoms.contains("selector"))
        XCTAssertTrue(loweredAtoms.contains("mod1"))
    }

    func testManifestIncludesModulatorsAndDestinations() throws {
        let source = """
        (def mod1 (in 5 @name mod1 @modulator 1))
        (def mod2 (in 6 @name mod2 @modulator 2))
        (param cutoff @default 2400 @min 60 @max 12000 @unit Hz @mod true @mod-mode additive @mod-depth-min -6000 @mod-depth-max 6000)
        (out (mod cutoff) 1 @name audio)
        """

        let evaluator = LispEvaluator()
        let lowered = try lowerModulation(in: parseSource(source))
        try evaluator.evaluate(nodes: lowered)

        let graph = LazyGraphContext.current
        for output in evaluator.outputs {
            graph.addOutput(output.signal, channel: output.channel)
        }

        let compilation = try graph.compileOnly(frameCount: 64, voiceCount: 1)
        let compilerResult = CompilerResult(
            dylibPath: "",
            cSourcePath: "",
            compilationResult: compilation,
            cSource: ""
        )
        let options = CompilerOptions(
            outputDir: ".",
            name: "patch",
            sampleRate: 48_000,
            maxFrames: 64,
            voiceCount: 1,
            debug: false
        )

        let manifest = generateManifest(
            compilerResult: compilerResult,
            evaluator: evaluator,
            options: options
        )

        XCTAssertEqual(manifest.modulators.count, 2)
        XCTAssertEqual(manifest.modulators.map(\.slot), [1, 2])
        XCTAssertEqual(manifest.modulators.map(\.inputChannel), [4, 5])

        let hiddenNames = Set(manifest.params.compactMap { $0.hidden == true ? $0.name : nil })
        XCTAssertTrue(hiddenNames.contains("__mod__cutoff__source"))
        XCTAssertTrue(hiddenNames.contains("__mod__cutoff__depth"))

        XCTAssertEqual(manifest.modDestinations.count, 1)
        let destination = try XCTUnwrap(manifest.modDestinations.first)
        XCTAssertEqual(destination.name, "cutoff")
        XCTAssertEqual(destination.mode, "additive")
        XCTAssertEqual(destination.min, 60)
        XCTAssertEqual(destination.max, 12000)
        XCTAssertEqual(destination.depthMin, -6000)
        XCTAssertEqual(destination.depthMax, 6000)
    }

    func testPercentIsModuloOperator() throws {
        let evaluator = LispEvaluator()
        try evaluator.evaluate(nodes: parseSource("""
        (def x (% 7 3))
        """))

        guard case .float(let result)? = evaluator.definitions["x"] else {
            return XCTFail("expected float result")
        }
        XCTAssertEqual(result, 1)
    }

    func testSelectorOperatorEvaluates() throws {
        let evaluator = LispEvaluator()
        try evaluator.evaluate(nodes: parseSource("""
        (def x (selector 2 10 20 30))
        """))

        guard case .signal(let signal)? = evaluator.definitions["x"] else {
            return XCTFail("expected signal result")
        }

        let values = try signal.realize(frames: 1)
        XCTAssertEqual(values.count, 1)
        XCTAssertEqual(values[0], 20, accuracy: 0.0001)
    }

    func testPowSupportsFloatBaseSignalExponent() throws {
        DGenConfig.sampleRate = 4
        LazyGraphContext.reset()

        let evaluator = LispEvaluator()
        try evaluator.evaluate(nodes: parseSource("""
        (def x (pow 2 (phasor 1)))
        """))

        guard case .signal(let signal)? = evaluator.definitions["x"] else {
            return XCTFail("expected signal result")
        }

        let values = try signal.realize(frames: 4)
        XCTAssertEqual(values[0], 1.0, accuracy: 0.0001)
        XCTAssertEqual(values[1], Float(Foundation.pow(2.0, 0.25)), accuracy: 0.0001)
        XCTAssertEqual(values[2], Float(Foundation.pow(2.0, 0.5)), accuracy: 0.0001)
        XCTAssertEqual(values[3], Float(Foundation.pow(2.0, 0.75)), accuracy: 0.0001)
    }

    func testPowSupportsFloatBaseTensorExponent() throws {
        let evaluator = LispEvaluator()
        try evaluator.evaluate(nodes: parseSource("""
        (def t (pow 2 (full [2] 3)))
        (def x (peek t 0))
        (def y (peek t 1))
        """))

        guard case .signal(let x)? = evaluator.definitions["x"],
              case .signal(let y)? = evaluator.definitions["y"] else {
            return XCTFail("expected peek results as signals")
        }

        XCTAssertEqual(try x.realize(frames: 1)[0], 8.0, accuracy: 0.0001)
        XCTAssertEqual(try y.realize(frames: 1)[0], 8.0, accuracy: 0.0001)
    }

    func testPowSupportsSignalTensorCombinations() throws {
        DGenConfig.sampleRate = 4
        LazyGraphContext.reset()

        let evaluator = LispEvaluator()
        try evaluator.evaluate(nodes: parseSource("""
        (def a (pow (phasor (full [2] 1)) 2))
        (def b (pow 2 (phasor (full [2] 1))))
        (def c (pow (phasor 1) (full [2] 2)))
        """))

        guard case .signalTensor(let a)? = evaluator.definitions["a"],
              case .signalTensor(let b)? = evaluator.definitions["b"],
              case .signalTensor(let c)? = evaluator.definitions["c"] else {
            return XCTFail("expected signalTensor results")
        }

        let aValues = try a.realize(frames: 2)
        XCTAssertEqual(aValues.count, 4)
        for (actual, expected) in zip(aValues.sorted(), [Float(0.0), 0.0, 0.0625, 0.0625]) {
            XCTAssertEqual(actual, expected, accuracy: 0.0001)
        }

        let bValues = try b.realize(frames: 2)
        XCTAssertEqual(bValues.count, 4)
        for (actual, expected) in zip(
            bValues.sorted(),
            [Float(1.0), 1.0, Float(Foundation.pow(2.0, 0.25)), Float(Foundation.pow(2.0, 0.25))]
        ) {
            XCTAssertEqual(actual, expected, accuracy: 0.0001)
        }

        let cValues = try c.realize(frames: 2)
        XCTAssertEqual(cValues.count, 4)
        for (actual, expected) in zip(cValues.sorted(), [Float(0.0), 0.0, 0.0625, 0.0625]) {
            XCTAssertEqual(actual, expected, accuracy: 0.0001)
        }
    }

    func testModRequiresModulatableParameter() throws {
        XCTAssertThrowsError(try lowerModulation(in: parseSource("""
        (param cutoff @default 1000 @min 20 @max 12000)
        (out (mod cutoff) 1)
        """))) { error in
            guard let lispError = error as? LispError else {
                return XCTFail("unexpected error: \(error)")
            }
            XCTAssertTrue(lispError.message.contains("not declared with @mod true"))
        }
    }
}
