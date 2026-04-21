import Foundation

extension LazyOp {
  func emitPhaseVocoderPitchShift(
    b: IRBuilder, ctx: IRContext, g: Graph, node: Node,
    inputs: [Lazy], nodeId: NodeID
  ) throws {
    guard case .phaseVocoderPitchShift(
      let N, let hopSize,
      let prevAnalysisPhaseCell,
      let synthPhaseCell,
      let tempMagCell,
      let tempOmegaCell,
      let initCell,
      let reOutCell,
      let imOutCell
    ) = self else { return }

    guard inputs.count == 4 else {
      throw DGenError.insufficientInputs(
        operator: "phaseVocoderPitchShift", expected: 4, actual: inputs.count)
    }
    guard let reTensorId = g.nodeToTensor[node.inputs[0]],
      let reTensor = g.tensors[reTensorId],
      let imTensorId = g.nodeToTensor[node.inputs[1]],
      let imTensor = g.tensors[imTensorId]
    else {
      throw DGenError.tensorError(
        op: "phaseVocoderPitchShift", reason: "inputs 0/1 must be tensor nodes")
    }

    let ratioInput = b.value(inputs[2])
    let counter = b.value(inputs[3])
    let zero = b.constant(0.0)
    let one = b.constant(1.0)
    let tau = b.constant(2.0 * Float.pi)
    let halfTau = b.constant(Float.pi)
    let omegaScale = b.constant(2.0 * Float.pi * Float(hopSize) / Float(N))
    let nMinusOne = b.constant(Float(N - 1))
    let nMinusOneInt = b.intConstant(N - 1)
    let ratioMin = b.constant(1e-4)
    let ratio = b.max(ratioInput, ratioMin)
    let initValue = b.memoryRead(initCell, zero)

    func principalArg(_ expr: Expr) -> Expr {
      let wrapped = expr - tau * b.round(expr / tau)
      // Keep the result inside [-pi, pi] after roundf edge cases.
      let over = wrapped > halfTau
      let under = wrapped < b.neg(halfTau)
      let wrappedHigh = b.gswitch(over, wrapped - tau, wrapped)
      return b.gswitch(under, wrappedHigh + tau, wrappedHigh)
    }

    b.if_(counter == zero) {
      // Analysis: current magnitude and true per-bin angular velocity.
      b.loop(N) { i in
        let reVal = b.tensorRead(reTensor, flatIdx: i, shape: [N])
        let imVal = b.tensorRead(imTensor, flatIdx: i, shape: [N])
        let mag = b.sqrt(reVal * reVal + imVal * imVal)
        let phase = b.atan2(imVal, reVal)
        let prevPhase = b.memoryRead(prevAnalysisPhaseCell, i)
        let omegaTarget = b.cast(i, to: .float) * omegaScale
        let delta = principalArg(phase - prevPhase - omegaTarget)
        let omegaActual = omegaTarget + delta

        _ = b.memoryWrite(tempMagCell, i, mag)
        _ = b.memoryWrite(tempOmegaCell, i, omegaActual)
        _ = b.memoryWrite(prevAnalysisPhaseCell, i, phase)
      }

      // Resynthesis: output-driven bin remap with linear interpolation in
      // source-bin space. For ratio == 1 this is identity; for other ratios it
      // moves energy to destination bins while preserving instantaneous
      // frequency tracking.
      b.loop(N) { k in
        let kFloat = b.cast(k, to: .float)
        let srcPos = kFloat / ratio
        let inRange = srcPos <= nMinusOne

        let srcPosClamped = b.gswitch(inRange, srcPos, nMinusOne)
        let srcFloorF = b.floor(srcPosClamped)
        let srcFloor = b.cast(srcFloorF, to: .int)
        let srcCeil = b.gswitch(srcFloor < nMinusOneInt, srcFloor + b.intConstant(1), srcFloor)
        let frac = srcPosClamped - srcFloorF

        let mag0 = b.memoryRead(tempMagCell, srcFloor)
        let mag1 = b.memoryRead(tempMagCell, srcCeil)
        let omega0 = b.memoryRead(tempOmegaCell, srcFloor)
        let omega1 = b.memoryRead(tempOmegaCell, srcCeil)
        let phase0 = b.memoryRead(prevAnalysisPhaseCell, srcFloor)
        let phase1 = b.memoryRead(prevAnalysisPhaseCell, srcCeil)

        let interpMag = b.mix(mag0, mag1, frac)
        let interpOmega = b.mix(omega0, omega1, frac) * ratio
        let interpPhase = principalArg(b.mix(phase0, phase1, frac))

        let prevSynth = b.memoryRead(synthPhaseCell, k)
        let advancedPhase = principalArg(prevSynth + interpOmega)
        let synthPhase = b.gswitch(initValue == zero, interpPhase, advancedPhase)
        let outMag = b.gswitch(inRange, interpMag, zero)
        let outRe = outMag * b.cos(synthPhase)
        let outIm = outMag * b.sin(synthPhase)

        _ = b.memoryWrite(synthPhaseCell, k, synthPhase)
        _ = b.memoryWrite(reOutCell, k, outRe)
        _ = b.memoryWrite(imOutCell, k, outIm)
      }

      _ = b.memoryWrite(initCell, zero, one)
    }

    ctx.values[nodeId] = .empty
  }
}
