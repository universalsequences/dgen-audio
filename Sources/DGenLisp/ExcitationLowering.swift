// ExcitationLowering.swift — drive instrument inlets per spec §6.
//
// The lazy-graph runtime renders with silent audio inputs, so `(in ...)`
// inlets are rewritten at the AST level into the v1 excitation convention
// before evaluation:
//
//   in @name trigger  ->  (click)                        single trigger at t=0
//   in @name gate     ->  sample-counter < gate_frames   hard-coded gate window
//   in @name pitch    ->  <pitch_hz>                     frozen CPU estimate
//   in @name velocity ->  1.0
//   in @modulator N   ->  left as an input inlet         no DAW modulation
//
// Modulator inlets are deliberately NOT rewritten: the lazy runtime
// renders input channels as silence, which is exactly "no modulation",
// and the modulated-param machinery requires real input signals (it
// traps on constants).
//
// Inlets the convention cannot drive are reported (plan.unsupported) —
// training a patch whose inputs would silently read zeros is a wrong
// answer, not a default.

import Foundation

enum ExcitationLowering {
    struct Rewrite {
        let nodes: [ASTNode]
        /// `(in ...)` nodes that could not be mapped to the excitation
        /// convention, as "in#<channel-or-name>" identifiers.
        let undriven: [String]
    }

    static func drive(nodes: [ASTNode], pitchHz: Double, gateFrames: Int) -> Rewrite {
        var undriven: [String] = []
        let rewritten = nodes.map { rewrite($0, pitchHz: pitchHz, gateFrames: gateFrames, undriven: &undriven) }
        return Rewrite(nodes: rewritten, undriven: undriven)
    }

    private static func rewrite(
        _ node: ASTNode, pitchHz: Double, gateFrames: Int, undriven: inout [String]
    ) -> ASTNode {
        guard case .list(let elements) = node else { return node }
        if case .atom("in") = elements.first ?? .atom("") {
            return replacementForInput(
                elements, pitchHz: pitchHz, gateFrames: gateFrames, undriven: &undriven)
        }
        return .list(
            elements.map { rewrite($0, pitchHz: pitchHz, gateFrames: gateFrames, undriven: &undriven) })
    }

    private static func replacementForInput(
        _ elements: [ASTNode], pitchHz: Double, gateFrames: Int, undriven: inout [String]
    ) -> ASTNode {
        var name: String?
        var channel: String = "?"
        var isModulator = false
        var i = 1
        while i < elements.count {
            if case .atom(let a) = elements[i] {
                if a == "@name", i + 1 < elements.count, case .atom(let v) = elements[i + 1] {
                    name = v
                    i += 2
                    continue
                }
                if a == "@modulator" {
                    isModulator = true
                    i += 2
                    continue
                }
                if !a.hasPrefix("@"), channel == "?" { channel = a }
            }
            i += 1
        }
        // DAW modulation sources stay as (silent) input inlets: the job
        // fits the un-modulated voice, matching a one-shot target.
        if isModulator {
            return .list(elements)
        }
        switch name {
        case "trigger":
            return .list([.atom("click")])
        case "gate":
            // 1 while the sample counter is below gate_frames, else 0.
            return .list([
                .atom("lt"),
                .list([.atom("accum"), .atom("1.0"), .atom("0.0"), .atom("0.0"), .atom("1000000000.0")]),
                .atom("\(Float(gateFrames))"),
            ])
        case "pitch":
            return .atom("\(pitchHz)")
        case "velocity":
            return .atom("1.0")
        default:
            undriven.append("in#\(name ?? channel)")
            return .list(elements)
        }
    }

    /// Remove DAW modulation from the patch before lowerModulation:
    /// training always runs with silent modulators, where `(mod x)` == `x`
    /// exactly, so the modulated-param machinery (whose generated kernels
    /// currently miscompile in the training pipeline) is dropped rather
    /// than trained-through. Strips `@mod*` attributes from params and
    /// unwraps 2-element `(mod name)` forms.
    static func stripModulation(nodes: [ASTNode]) -> [ASTNode] {
        nodes.map(stripModulationNode)
    }

    private static let modAttributes: Set<String> = [
        "@mod", "@mod-mode", "@mod-depth-min", "@mod-depth-max",
    ]

    private static func stripModulationNode(_ node: ASTNode) -> ASTNode {
        guard case .list(let elements) = node else { return node }
        // (mod name) — same predicate ModulationLowering uses.
        if elements.count == 2, case .atom("mod") = elements[0], case .atom = elements[1] {
            return elements[1]
        }
        if case .atom("param") = elements.first ?? .atom("") {
            var kept: [ASTNode] = []
            var i = 0
            while i < elements.count {
                if case .atom(let a) = elements[i], modAttributes.contains(a) {
                    i += 2  // drop the attribute and its value
                    continue
                }
                kept.append(elements[i])
                i += 1
            }
            return .list(kept)
        }
        return .list(elements.map(stripModulationNode))
    }

    /// Serialize AST back to lisp source (for lowered.lisp).
    static func printAST(_ node: ASTNode) -> String {
        switch node {
        case .atom(let a): return a
        case .list(let elements):
            return "(" + elements.map(printAST).joined(separator: " ") + ")"
        }
    }
}
