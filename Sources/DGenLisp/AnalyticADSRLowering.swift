// AnalyticADSRLowering.swift — train-only closed-form ADSR for the known
// one-shot excitation used by TrainPlanner. Render/checkpoint ASTs never pass
// through this lowering.

import Foundation

enum AnalyticADSRLowering {
    private static let macroName = "__dgen_train_analytic_adsr"
    private static let counterName = "__dgen_train_sample_index"

    /// Replaces ADSR calls with a history-free envelope in absolute sample
    /// time. The one shared accumulator replaces the three history cells in
    /// every ordinary ADSR instance. A one-sample soft stage blend gives the
    /// duration parameters a gradient at stage boundaries; `+ 1` sample
    /// duration floors retain a non-zero attack gradient when attack_ms is 0.
    static func lower(nodes: [ASTNode], gateFrames: Int) throws -> [ASTNode] {
        var didRewrite = false
        let body = nodes.map { rewrite($0, didRewrite: &didRewrite) }
        guard didRewrite else { return nodes }

        // x is one-based to match the real ADSR's first-frame attack update.
        // k = ln(1000), matching the real macro's settle-time exponentials.
        let preamble = try parseSource("""
            (def \(counterName) (accum 1.0 0.0 0.0 1000000000.0))
            (defmacro \(macroName) (attack_ms decay_ms sustain release_ms)
              (def __adsr_x (+ \(counterName) 1.0))
              (def __adsr_sr samplerate)
              (def __adsr_k 6.907755)
              (def __adsr_attack_samples (+ 1.0 (* attack_ms 0.001 __adsr_sr)))
              (def __adsr_decay_samples (+ 1.0 (* decay_ms 0.001 __adsr_sr)))
              (def __adsr_release_samples (+ 1.0 (* release_ms 0.001 __adsr_sr)))

              (def __adsr_attack_level
                (min 1.0 (/ __adsr_x __adsr_attack_samples)))
              (def __adsr_decay_elapsed
                (max 0.0 (- __adsr_x __adsr_attack_samples)))
              (def __adsr_decay_level
                (+ sustain
                   (* (- 1.0 sustain)
                      (exp (/ (* -1.0 __adsr_k __adsr_decay_elapsed)
                              __adsr_decay_samples)))))
              (def __adsr_attack_weight
                (clip (+ 0.5 (* 0.5 (- __adsr_attack_samples __adsr_x))) 0.0 1.0))
              (def __adsr_held
                (+ (* __adsr_attack_weight __adsr_attack_level)
                   (* (- 1.0 __adsr_attack_weight) __adsr_decay_level)))

              (def __adsr_gate_x \(Float(max(0, gateFrames))))
              (def __adsr_anchor_attack
                (min 1.0 (/ __adsr_gate_x __adsr_attack_samples)))
              (def __adsr_anchor_decay_elapsed
                (max 0.0 (- __adsr_gate_x __adsr_attack_samples)))
              (def __adsr_anchor_decay
                (+ sustain
                   (* (- 1.0 sustain)
                      (exp (/ (* -1.0 __adsr_k __adsr_anchor_decay_elapsed)
                              __adsr_decay_samples)))))
              (def __adsr_anchor_attack_weight
                (clip
                  (+ 0.5 (* 0.5 (- __adsr_attack_samples __adsr_gate_x)))
                  0.0 1.0))
              (def __adsr_anchor
                (+ (* __adsr_anchor_attack_weight __adsr_anchor_attack)
                   (* (- 1.0 __adsr_anchor_attack_weight) __adsr_anchor_decay)))
              (def __adsr_release_elapsed
                (max 0.0 (- __adsr_x __adsr_gate_x)))
              (def __adsr_released
                (* __adsr_anchor
                   (exp (/ (* -1.0 __adsr_k __adsr_release_elapsed)
                           __adsr_release_samples))))
              (def __adsr_gate_weight
                (clip
                  (+ 0.5 (* 0.5 (- (+ __adsr_gate_x 0.5) __adsr_x)))
                  0.0 1.0))
              (+ (* __adsr_gate_weight __adsr_held)
                 (* (- 1.0 __adsr_gate_weight) __adsr_released)))
            """)
        return preamble + body
    }

    private static func rewrite(_ node: ASTNode, didRewrite: inout Bool) -> ASTNode {
        guard case .list(let elements) = node, !elements.isEmpty else { return node }

        // Keep the real declaration intact. Only call sites are redirected.
        if elements.count >= 2,
           case .atom("defmacro") = elements[0],
           case .atom("adsr") = elements[1]
        {
            return node
        }

        let children = elements.map { rewrite($0, didRewrite: &didRewrite) }
        guard children.count == 7, case .atom("adsr") = children[0] else {
            return .list(children)
        }
        didRewrite = true
        // gate and trigger are intentionally omitted: TrainPlanner has already
        // replaced them with its known single-trigger, fixed gate excitation.
        return .list([.atom(macroName)] + Array(children[3...6]))
    }
}
