import DGen
import Foundation

// MARK: - Extra Math

public func log10(_ x: Tensor) -> Tensor {
  let nodeId = x.graph.node(.log10, [x.nodeId])
  return Tensor(nodeId: nodeId, graph: x.graph, shape: x.shape, requiresGrad: x.requiresGrad)
}

public func log10(_ x: Signal) -> Signal {
  let nodeId = x.graph.node(.log10, [x.nodeId])
  return Signal(nodeId: nodeId, graph: x.graph, requiresGrad: x.requiresGrad)
}

public func log10(_ x: SignalTensor) -> SignalTensor {
  let nodeId = x.graph.node(.log10, [x.nodeId])
  return SignalTensor(nodeId: nodeId, graph: x.graph, shape: x.shape, requiresGrad: x.requiresGrad)
}

public func atan2(_ y: Tensor, _ x: Tensor) -> Tensor {
  let nodeId = y.graph.node(.atan2, [y.nodeId, x.nodeId])
  return Tensor(
    nodeId: nodeId, graph: y.graph, shape: broadcastShape(y.shape, x.shape),
    requiresGrad: y.requiresGrad || x.requiresGrad)
}

public func atan2(_ y: Signal, _ x: Signal) -> Signal {
  let nodeId = y.graph.node(.atan2, [y.nodeId, x.nodeId])
  return Signal(nodeId: nodeId, graph: y.graph, requiresGrad: y.requiresGrad || x.requiresGrad)
}

public func atan2(_ y: SignalTensor, _ x: SignalTensor) -> SignalTensor {
  let nodeId = y.graph.node(.atan2, [y.nodeId, x.nodeId])
  return SignalTensor(
    nodeId: nodeId, graph: y.graph, shape: broadcastShape(y.shape, x.shape),
    requiresGrad: y.requiresGrad || x.requiresGrad)
}

// MARK: - Spectral Helpers

public func hann(_ n: Int) -> Tensor {
  precondition(n > 0, "hann: N must be positive")
  let scale = 2.0 * Float.pi / Float(n)
  let data = (0..<n).map { i in
    0.5 - 0.5 * Foundation.cos(scale * Float(i))
  }
  return Tensor(data)
}

public func polarFFT(_ re: Tensor, _ im: Tensor) -> (mag: Tensor, phase: Tensor) {
  let mag = sqrt(re * re + im * im)
  let phase = DGenLazy.atan2(im, re)
  return (mag, phase)
}

public func polarFFT(_ re: SignalTensor, _ im: SignalTensor) -> (mag: SignalTensor, phase: SignalTensor) {
  let mag = sqrt(re * re + im * im)
  let phase = DGenLazy.atan2(im, re)
  return (mag, phase)
}

public func rectFFT(_ mag: Tensor, _ phase: Tensor) -> (re: Tensor, im: Tensor) {
  return (mag * cos(phase), mag * sin(phase))
}

public func rectFFT(_ mag: SignalTensor, _ phase: SignalTensor) -> (re: SignalTensor, im: SignalTensor) {
  return (mag * cos(phase), mag * sin(phase))
}

public func complexMul(_ ar: Tensor, _ ai: Tensor, _ br: Tensor, _ bi: Tensor) -> (re: Tensor, im: Tensor) {
  return (ar * br - ai * bi, ar * bi + ai * br)
}

public func complexMul(
  _ ar: SignalTensor, _ ai: SignalTensor, _ br: Tensor, _ bi: Tensor
) -> (re: SignalTensor, im: SignalTensor) {
  return (ar * br - ai * bi, ar * bi + ai * br)
}

public func complexMul(
  _ ar: SignalTensor, _ ai: SignalTensor, _ br: SignalTensor, _ bi: SignalTensor
) -> (re: SignalTensor, im: SignalTensor) {
  return (ar * br - ai * bi, ar * bi + ai * br)
}

public func complexConj(_ re: Tensor, _ im: Tensor) -> (re: Tensor, im: Tensor) {
  return (re, -im)
}

public func complexConj(_ re: SignalTensor, _ im: SignalTensor) -> (re: SignalTensor, im: SignalTensor) {
  return (re, -im)
}

// MARK: - FFT Backends

public func acceleratedFFT(_ input: Tensor, N: Int) -> (re: Tensor, im: Tensor) {
  let (reId, imId) = input.graph.graph.acceleratedFFT(input.nodeId, N: N)
  return (
    Tensor(nodeId: reId, graph: input.graph, shape: [N], requiresGrad: input.requiresGrad),
    Tensor(nodeId: imId, graph: input.graph, shape: [N], requiresGrad: input.requiresGrad)
  )
}

public func acceleratedFFT(_ input: SignalTensor, N: Int) -> (re: SignalTensor, im: SignalTensor) {
  let (reId, imId) = input.graph.graph.acceleratedFFT(input.nodeId, N: N)
  return (
    SignalTensor(nodeId: reId, graph: input.graph, shape: [N], requiresGrad: input.requiresGrad),
    SignalTensor(nodeId: imId, graph: input.graph, shape: [N], requiresGrad: input.requiresGrad)
  )
}

public func acceleratedIFFT(_ re: Tensor, _ im: Tensor, N: Int) -> Tensor {
  let nodeId = re.graph.graph.acceleratedIFFT(re.nodeId, im.nodeId, N: N)
  return Tensor(nodeId: nodeId, graph: re.graph, shape: [N], requiresGrad: re.requiresGrad || im.requiresGrad)
}

public func acceleratedIFFT(_ re: SignalTensor, _ im: SignalTensor, N: Int) -> SignalTensor {
  let nodeId = re.graph.graph.acceleratedIFFT(re.nodeId, im.nodeId, N: N)
  return SignalTensor(
    nodeId: nodeId, graph: re.graph, shape: [N], requiresGrad: re.requiresGrad || im.requiresGrad)
}

// MARK: - Hop and Noise

public func tensorNoise(size: Int, hop: Int? = nil) -> SignalTensor {
  let graph = LazyGraphContext.current
  let nodeId = graph.graph.noise(size: size, hopSize: hop)
  return SignalTensor(nodeId: nodeId, graph: graph, shape: [size], requiresGrad: false)
}

extension Signal {
  public func hopHold(hop: Int) -> Signal {
    let nodeId = graph.graph.hopHold(self.nodeId, hopSize: hop)
    return Signal(nodeId: nodeId, graph: graph, requiresGrad: requiresGrad)
  }
}

extension SignalTensor {
  public func hopHold(hop: Int) -> SignalTensor {
    let nodeId = graph.graph.hopHold(self.nodeId, hopSize: hop)
    return SignalTensor(nodeId: nodeId, graph: graph, shape: shape, requiresGrad: requiresGrad)
  }
}

extension Tensor {
  public func hopHold(hop: Int) -> SignalTensor {
    let nodeId = graph.graph.hopHold(self.nodeId, hopSize: hop)
    return SignalTensor(nodeId: nodeId, graph: graph, shape: shape, requiresGrad: requiresGrad)
  }
}

// MARK: - Tensor Ops

extension TensorOps {
  public func conv1d(_ kernel: Tensor) -> Self {
    guard shape.count == 1, kernel.shape.count == 1 else {
      fatalError("conv1d requires 1D input and 1D kernel tensor")
    }
    let nodeId = graph.graph.n(.conv1d(kernel.shape[0]), self.nodeId, kernel.nodeId)
    return Self(
      _view: nodeId, graph: graph, shape: shape,
      requiresGrad: requiresGrad || kernel.requiresGrad)
  }
}

// MARK: - Stateful Spectral Ops

extension SignalTensor {
  public func spectrumDelay(N: Int, hops: Int, hop: Int) -> SignalTensor {
    let nodeId = graph.graph.spectrumDelay(self.nodeId, N: N, hops: hops, hopSize: hop)
    return SignalTensor(nodeId: nodeId, graph: graph, shape: [N], requiresGrad: requiresGrad)
  }

  public func spectrumDelayMod(delay: Signal, N: Int, maxHops: Int, hop: Int) -> SignalTensor {
    let nodeId = graph.graph.spectrumDelayMod(self.nodeId, delay: delay.nodeId, N: N, maxHops: maxHops, hopSize: hop)
    return SignalTensor(
      nodeId: nodeId, graph: graph, shape: [N], requiresGrad: requiresGrad || delay.requiresGrad)
  }
}

public func phaseVocoder(
  _ re: SignalTensor, _ im: SignalTensor, ratio: Signal, N: Int, hop: Int
) -> (re: SignalTensor, im: SignalTensor) {
  let (reId, imId) = re.graph.graph.phaseVocoder(re.nodeId, im.nodeId, pitchRatio: ratio.nodeId, N: N, hopSize: hop)
  return (
    SignalTensor(nodeId: reId, graph: re.graph, shape: [N], requiresGrad: re.requiresGrad || ratio.requiresGrad),
    SignalTensor(nodeId: imId, graph: re.graph, shape: [N], requiresGrad: im.requiresGrad || ratio.requiresGrad)
  )
}

public func partitionedSpectralMAC(
  _ xRe: SignalTensor, _ xIm: SignalTensor, _ irRe: Tensor, _ irIm: Tensor, N: Int
) -> (re: SignalTensor, im: SignalTensor) {
  let irSize = irRe.shape.reduce(1, *)
  precondition(irSize % N == 0 && irSize >= N, "partitionedSpectralMAC: IR size must be a positive multiple of N")
  let k = irSize / N
  let hop = xRe.graph.graph.nodeHopRate[xRe.nodeId]?.0 ?? max(1, N / 2)
  let (reId, imId) = xRe.graph.graph.partitionedSpectralConvolve(
    xRe.nodeId, xIm.nodeId, irRe.nodeId, irIm.nodeId, K: k, N: N, hopSize: hop)
  return (
    SignalTensor(nodeId: reId, graph: xRe.graph, shape: [N], requiresGrad: xRe.requiresGrad),
    SignalTensor(nodeId: imId, graph: xRe.graph, shape: [N], requiresGrad: xIm.requiresGrad)
  )
}

public func partitionIR(_ ir: Tensor, N: Int, hop: Int) -> (re: Tensor, im: Tensor) {
  precondition(N > 0 && (N & (N - 1)) == 0, "partitionIR: N must be a power of 2")
  precondition(hop > 0 && hop <= N, "partitionIR: hop must be in 1...N")
  guard let data = ir.getData() else {
    fatalError("partitionIR requires a tensor with resident data")
  }
  let k = max(1, Int(ceil(Double(data.count) / Double(hop))))
  var allRe = [Float]()
  var allIm = [Float]()
  allRe.reserveCapacity(k * N)
  allIm.reserveCapacity(k * N)
  for part in 0..<k {
    var re = [Float](repeating: 0, count: N)
    var im = [Float](repeating: 0, count: N)
    let start = part * hop
    let end = min(start + hop, data.count)
    if start < end {
      for i in start..<end { re[i - start] = data[i] }
    }
    radix2FFT(re: &re, im: &im)
    allRe.append(contentsOf: re)
    allIm.append(contentsOf: im)
  }
  return (Tensor(allRe), Tensor(allIm))
}

public func partitionedConvolve(_ input: Signal, _ ir: Tensor, N: Int, hop: Int, gain: Float = 1.0) -> Signal {
  let win = hann(N)
  let buffered = input.buffer(size: N, hop: hop).reshape([N])
  let windowed = buffered * win
  let (xRe, xIm) = acceleratedFFT(windowed, N: N)
  let (irRe, irIm) = partitionIR(ir, N: N, hop: hop)
  let (yRe, yIm) = partitionedSpectralMAC(xRe, xIm, irRe, irIm, N: N)
  let td = acceleratedIFFT(yRe, yIm, N: N)
  return (td * win * gain).overlapAdd(hop: hop)
}

private func radix2FFT(re: inout [Float], im: inout [Float]) {
  let n = re.count
  precondition(n == im.count && n > 0 && (n & (n - 1)) == 0)

  var j = 0
  for i in 1..<n {
    var bit = n >> 1
    while j & bit != 0 {
      j ^= bit
      bit >>= 1
    }
    j ^= bit
    if i < j {
      re.swapAt(i, j)
      im.swapAt(i, j)
    }
  }

  var len = 2
  while len <= n {
    let angle = -2.0 * Float.pi / Float(len)
    let wLenRe = Foundation.cos(angle)
    let wLenIm = Foundation.sin(angle)
    for i in stride(from: 0, to: n, by: len) {
      var wRe: Float = 1
      var wIm: Float = 0
      for offset in 0..<(len / 2) {
        let uRe = re[i + offset]
        let uIm = im[i + offset]
        let vRe = re[i + offset + len / 2] * wRe - im[i + offset + len / 2] * wIm
        let vIm = re[i + offset + len / 2] * wIm + im[i + offset + len / 2] * wRe
        re[i + offset] = uRe + vRe
        im[i + offset] = uIm + vIm
        re[i + offset + len / 2] = uRe - vRe
        im[i + offset + len / 2] = uIm - vIm
        let nextRe = wRe * wLenRe - wIm * wLenIm
        let nextIm = wRe * wLenIm + wIm * wLenRe
        wRe = nextRe
        wIm = nextIm
      }
    }
    len <<= 1
  }
}

