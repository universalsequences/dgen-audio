#if canImport(Accelerate)
import Accelerate
#else
import DGenLazy  // portable vDSP stand-ins (PortableAccelerate.swift)
#endif
import DGenTrainProtocol
import Foundation

/// Independent CPU MR-STFT score used only to rank multistart lanes. This is
/// deliberately outside the training graph: the GPU batched spectral loss
/// reduces over lanes, while selection needs one score per rendered lane.
final class TrainSpectralScorer {
    struct Plan {
        let size: Int
        let hann: [Float]
        let scale: Float
        let setup: vDSP_DFT_Setup
        let starts: [Int]
        let bins: Int
        var targetLog: [Float]
    }

    private var plans: [Plan] = []
    private let epsilon: Float

    init(target: [Float], windows: [Int], epsilon: Float = 1e-3) throws {
        self.epsilon = epsilon
        for size in windows where size <= target.count {
            guard let setup = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(size), .FORWARD) else {
                throw TrainProtocolError("vDSP DFT setup failed for window \(size)")
            }
            let hann = (0..<size).map {
                0.5 - 0.5 * Foundation.cos(2 * Float.pi * Float($0) / Float(size - 1))
            }
            let hop = max(1, size / 4)
            let starts = stride(from: 0, through: max(0, target.count - size), by: hop).map { $0 }
            var plan = Plan(
                size: size, hann: hann, scale: max(hann.reduce(0, +) / 2, 1e-12),
                setup: setup, starts: starts, bins: size / 2 + 1, targetLog: [])
            plan.targetLog = logSpectra(target, plan: plan)
            plans.append(plan)
        }
    }

    deinit {
        for plan in plans { vDSP_DFT_DestroySetup(plan.setup) }
    }

    private func logSpectra(_ signal: [Float], plan: Plan) -> [Float] {
        var input = [Float](repeating: 0, count: plan.size)
        let zeros = [Float](repeating: 0, count: plan.size)
        var real = zeros
        var imag = zeros
        var result = [Float](repeating: 0, count: plan.starts.count * plan.bins)
        for (frame, start) in plan.starts.enumerated() {
            signal.withUnsafeBufferPointer { samples in
                plan.hann.withUnsafeBufferPointer { window in
                    input.withUnsafeMutableBufferPointer { destination in
                        vDSP_vmul(
                            samples.baseAddress! + start, 1, window.baseAddress!, 1,
                            destination.baseAddress!, 1, vDSP_Length(plan.size))
                    }
                }
            }
            vDSP_DFT_Execute(plan.setup, input, zeros, &real, &imag)
            for bin in 0..<plan.bins {
                let magnitude = Foundation.sqrt(real[bin] * real[bin] + imag[bin] * imag[bin])
                    / plan.scale
                result[frame * plan.bins + bin] = Foundation.log(magnitude + epsilon)
            }
        }
        return result
    }

    func score(_ signal: [Float]) -> Float {
        var total: Float = 0
        for plan in plans {
            let actual = logSpectra(signal, plan: plan)
            var difference: Float = 0
            for index in actual.indices {
                difference += abs(actual[index] - plan.targetLog[index])
            }
            total += difference / Float(max(1, actual.count))
        }
        return total
    }
}
