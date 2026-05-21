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

        XCTAssertTrue(loweredAtoms.contains("__mod__cutoff__active"))
        XCTAssertTrue(loweredAtoms.contains("__mod__cutoff__depth__slot1"))
        XCTAssertTrue(loweredAtoms.contains("__mod__cutoff__resolved"))
        XCTAssertTrue(loweredAtoms.contains("__modulated-param"))
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
        XCTAssertTrue(hiddenNames.contains("__mod__cutoff__active"))
        XCTAssertTrue(hiddenNames.contains("__mod__cutoff__depth__slot1"))
        XCTAssertTrue(hiddenNames.contains("__mod__cutoff__depth__slot2"))
        let paramsByName = Dictionary(uniqueKeysWithValues: manifest.params.map { ($0.name, $0) })
        XCTAssertEqual(paramsByName["__mod__cutoff__active"]?.cellSpan, 1)
        XCTAssertEqual(paramsByName["__mod__cutoff__depth__slot1"]?.cellSpan, 1)
        XCTAssertEqual(paramsByName["__mod__cutoff__depth__slot2"]?.cellSpan, 1)

        XCTAssertEqual(manifest.modDestinations.count, 1)
        let destination = try XCTUnwrap(manifest.modDestinations.first)
        XCTAssertEqual(destination.name, "cutoff")
        XCTAssertEqual(destination.mode, "additive")
        XCTAssertEqual(destination.min, 60)
        XCTAssertEqual(destination.max, 12000)
        XCTAssertEqual(destination.depthMin, -6000)
        XCTAssertEqual(destination.depthMax, 6000)
        XCTAssertEqual(destination.depthLanes.map(\.slot), [1, 2])
    }

    func testManifestKeepsScalarParamCellSpanWhenBroadcastInSIMD() throws {
        let source = """
        (param gain @default 0.5 @min 0 @max 1)
        (out (* gain 0.25) 1)
        """

        let evaluator = LispEvaluator()
        try evaluator.evaluate(nodes: parseSource(source))

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

        let gain = try XCTUnwrap(manifest.params.first { $0.name == "gain" })
        XCTAssertEqual(gain.cellSpan, 1)
    }

    func testModulatableParamCellsDoNotOverlapLaterParamsWithADSRPreamble() throws {
        let source = """
        (def samplerate 48000.0)

        (defmacro adsr (gate_sig trigger_sig attack_ms decay_ms sustain release_ms)
          (make-history env)
          (make-history gate_hist)
          (make-history stage_hist)

          (def sr samplerate)
          (def env_time_scale 6.907755)
          (def reset_samples (* 0.003 sr))
          (def attack_samples (max 1.0 (* attack_ms 0.001 sr)))
          (def decay_samples (max 1.0 (* decay_ms 0.001 sr)))
          (def release_samples (max 1.0 (* release_ms 0.001 sr)))
          (def reset_coeff (- 1.0 (exp (/ (* -1.0 env_time_scale) reset_samples))))
          (def decay_coeff (- 1.0 (exp (/ (* -1.0 env_time_scale) decay_samples))))
          (def release_coeff (- 1.0 (exp (/ (* -1.0 env_time_scale) release_samples))))

          (def prev_env (read-history env))
          (def prev_gate (read-history gate_hist))
          (def prev_stage (read-history stage_hist))

          (def gate_on (gt gate_sig 0.5))
          (def gate_rising (* gate_on (lte prev_gate 0.5)))
          (def retrigger (max gate_rising trigger_sig))
          (def attack_stage 1.0)
          (def decay_stage 2.0)
          (def reset_stage 3.0)
          (def attack_done (gte prev_env 0.999))
          (def reset_done (lte prev_env 0.0001))

          (def stage_from_gate
            (gswitch gate_on
              (gswitch retrigger
                (gswitch (gt prev_env 0.0001) reset_stage attack_stage)
                prev_stage)
              0.0))

          (def stage
            (gswitch (eq stage_from_gate reset_stage)
              (gswitch reset_done attack_stage reset_stage)
              (gswitch attack_done
                (gswitch (eq stage_from_gate attack_stage) decay_stage stage_from_gate)
                stage_from_gate)))

          (def target
            (gswitch gate_on
              (gswitch (eq stage reset_stage)
                0.0
                (gswitch (eq stage attack_stage) 1.0 sustain))
              0.0))

          (def rate
            (gswitch gate_on
              (gswitch (eq stage reset_stage) reset_coeff decay_coeff)
              release_coeff))

          (def one_pole_level (+ prev_env (* rate (- target prev_env))))
          (def attack_level (+ prev_env (/ 1.0 attack_samples)))
          (def level_raw
            (gswitch (eq stage attack_stage)
              attack_level
              one_pole_level))
          (def level (clip level_raw 0 1))
          (write-history env level)
          (write-history gate_hist gate_sig)
          (write-history stage_hist stage)
          level)

        (defmacro op (input input2) (def phasor1 (phasor input)) (def mul1 (* phasor1 input2)) mul1)
        (def gate (in 1 @name gate))
        (def pitch (in 2 @name pitch))
        (def velocity (in 3 @name velocity))
        (def trigger (in 4 @name trigger))
        (def mod1 (in 5 @name mod1 @modulator 1))
        (def mod2 (in 6 @name mod2 @modulator 2))
        (def mod3 (in 7 @name mod3 @modulator 3))
        (def mod4 (in 8 @name mod4 @modulator 4))
        (def mod5 (in 9 @name mod5 @modulator 5))
        (def mod6 (in 10 @name mod6 @modulator 6))
        (def ext1 (in 11 @name ext1 @modulator 7))
        (def ext2 (in 12 @name ext2 @modulator 8))
        (def ext3 (in 13 @name ext3 @modulator 9))
        (def ext4 (in 14 @name ext4 @modulator 10))
        (param xout @default 1.0 @min 1.0 @max 2.0 @mod true @mod-mode additive)
        (def modulated1 (mod xout))
        (def op1 (op pitch modulated1))
        (param attack @default 5.0 @min 0.0 @max 1000.0 @unit ms)
        (param decay @default 120.0 @min 1.0 @max 2000.0 @unit ms)
        (param sustain @default 0.8 @min 0.0 @max 1.0)
        (param release @default 180.0 @min 1.0 @max 5000.0 @unit ms)
        (param gain @default 0.5 @min 0.0 @max 1.0 @mod true @mod-mode additive)
        (def env (adsr gate trigger attack decay sustain release))
        (def osc (scale op1 0.0 1.0 -1.0 1.0))
        (out (* osc env velocity (mod gain)) 1 @name audio)
        """

        let evaluator = LispEvaluator()
        let lowered = try lowerModulation(in: parseSource(source))
        try evaluator.evaluate(nodes: lowered)

        let graph = LazyGraphContext.current
        for output in evaluator.outputs {
            graph.addOutput(output.signal, channel: output.channel)
        }

        let compilation = try graph.compileOnly(frameCount: 128, voiceCount: 12)
        let compilerResult = CompilerResult(
            dylibPath: "",
            cSourcePath: "",
            compilationResult: compilation,
            cSource: compilation.kernels.map { $0.source }.joined(separator: "\n\n")
        )
        let options = CompilerOptions(
            outputDir: ".",
            name: "patch",
            sampleRate: 48_000,
            maxFrames: 128,
            voiceCount: 12,
            debug: false
        )

        let manifest = generateManifest(
            compilerResult: compilerResult,
            evaluator: evaluator,
            options: options
        )

        var ownerByCell: [Int: String] = [:]
        var collisions: [String] = []
        for param in manifest.params {
            for cell in param.cellId..<(param.cellId + param.cellSpan) {
                if let owner = ownerByCell[cell] {
                    collisions.append("memory[\(cell)] is shared by \(owner) and \(param.name)")
                } else {
                    ownerByCell[cell] = param.name
                }
            }
        }

        XCTAssertTrue(
            collisions.isEmpty,
            "DGenLisp manifest assigned overlapping param cells:\n\(collisions.joined(separator: "\n"))"
        )
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
