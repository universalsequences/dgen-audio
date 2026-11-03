import XCTest

@testable import DGenLazy

/// Input gradients through strided-window (`pool` / `windows`) views: overlapping windows must
/// scatter-add their gradients back into the positions they read. Checked against central
/// finite differences, including through two stacked im2col conv layers (the input grad of
/// layer 2 is the output grad of layer 1, so a wrong window backward corrupts layer 1's
/// kernel grad).
final class PoolGradientTests: XCTestCase {

  override func setUp() {
    super.setUp()
    LazyGraphContext.reset()
  }

  /// Deterministic values in [-1, 1].
  private func values(_ n: Int, seed: Int) -> [Float] {
    (0..<n).map { i in Float(Foundation.sin(Double(i * 7 + seed * 13) * 0.731)) }
  }

  /// Analytic grads of `build`'s scalar loss w.r.t. each named leaf, and central-difference
  /// grads of the same loss.
  private func check(
    leaves: [(shape: [Int], data: [Float])], tolerance: Float = 2e-3,
    _ build: ([Tensor]) -> Tensor, file: StaticString = #filePath, line: UInt = #line
  ) throws {
    LazyGraphContext.reset()
    let params = leaves.map { Tensor.param($0.shape, data: $0.data) }
    _ = try build(params).backward(frameCount: 1)
    let analytic = params.map { $0.grad?.getData() ?? [] }

    func lossAt(_ datas: [[Float]]) throws -> Float {
      LazyGraphContext.reset()
      let ts = zip(leaves, datas).map { Tensor($1).reshape($0.shape) }
      return try build(ts).realize()[0]
    }
    let h: Float = 1e-2
    for (k, leaf) in leaves.enumerated() {
      XCTAssertEqual(analytic[k].count, leaf.data.count, "leaf \(k) has no gradient", file: file, line: line)
      for i in 0..<leaf.data.count {
        var up = leaves.map(\.data), dn = up
        up[k][i] += h
        dn[k][i] -= h
        let fd = (try lossAt(up) - lossAt(dn)) / (2 * h)
        let ad = analytic[k][i]
        XCTAssertEqual(ad, fd, accuracy: tolerance * Swift.max(1, Swift.abs(fd)),
                       "leaf \(k)[\(i)]: analytic \(ad) vs fd \(fd)", file: file, line: line)
      }
    }
  }

  /// 1-D overlapping pool: every input position is read by up to k windows.
  func testPool1DOverlappingInputGrad() throws {
    let c = values(4 * 3, seed: 1)
    try check(leaves: [([9], values(9, seed: 2))]) { p in
      let w = p[0].pool(kernel: [3], stride: [2])  // [4, 3]
      return (w * w * Tensor(c).reshape([4, 3])).sum()
    }
  }

  /// Batched multi-channel pool over the last axis, stride 1.
  func testPoolBatchedStride1InputGrad() throws {
    let c = values(2 * 3 * 5 * 3, seed: 3)
    try check(leaves: [([2, 3, 7], values(42, seed: 4))]) { p in
      let w = p[0].pool(kernel: [3], stride: [1])  // [2, 3, 5, 3]
      return (w * Tensor(c).reshape([2, 3, 5, 3])).sum()
    }
  }

  /// Non-overlapping pool (stride == kernel) with a remainder the windows never read.
  func testPoolNonOverlappingWithRemainder() throws {
    let c = values(3 * 2, seed: 5)
    try check(leaves: [([7], values(7, seed: 6))]) { p in
      let w = p[0].pool(kernel: [2])  // [3, 2]; x[6] unread
      return (w * Tensor(c).reshape([3, 2])).sum()
    }
  }

  /// 2-D windows (the lowered conv2d path): input grad through overlapping 2-D windows.
  func testWindows2DInputGrad() throws {
    let c = values(3 * 3 * 2 * 3, seed: 7)
    try check(leaves: [([4, 5], values(20, seed: 8))]) { p in
      let w = p[0].windows([2, 3])  // [3, 3, 2, 3]
      return (w * Tensor(c).reshape([3, 3, 2, 3])).sum()
    }
  }

  /// conv2d input grad (kernel and input both learnable).
  func testConv2DInputAndKernelGrad() throws {
    try check(leaves: [([4, 5], values(20, seed: 9)), ([2, 2], values(4, seed: 10))]) { p in
      let y = p[0].conv2d(p[1])  // [3, 4]
      return (y * y).sum()
    }
  }

  /// Two stacked im2col conv1d layers (multi-channel, the second strided) with a tanh between.
  func testStackedConv1DGrads() throws {
    let cin = 2, t = 11, c1 = 3, k1 = 3, c2 = 2, k2 = 3, s2 = 2
    let t1 = t - k1 + 1  // 9
    let t2 = (t1 - k2) / s2 + 1  // 4
    let readout = values(t2 * c2, seed: 11)

    func conv(_ x: Tensor, cin: Int, k: Int, s: Int, w: Tensor) -> Tensor {
      // x [cin, T] → windows [cin, T', k] → [T', cin, k] → [T', cin·k] @ w [cin·k, cout]
      let win = x.pool(kernel: [k], stride: [s])
      let tp = win.shape[1]
      return win.transpose([1, 0, 2]).reshape([tp, cin * k]).matmul(w)
    }
    try check(leaves: [
      ([cin, t], values(cin * t, seed: 12)),
      ([cin * k1, c1], values(cin * k1 * c1, seed: 13).map { $0 * 0.5 }),
      ([c1 * k2, c2], values(c1 * k2 * c2, seed: 14).map { $0 * 0.5 }),
    ]) { p in
      let h1 = tanh(conv(p[0], cin: cin, k: k1, s: 1, w: p[1]))  // [t1, c1]
      let y = conv(h1.transpose([1, 0]), cin: c1, k: k2, s: s2, w: p[2])  // [t2, c2]
      return (y * Tensor(readout).reshape([t2, c2])).sum()
    }
  }
}

// MARK: - Window views over non-contiguous inputs, and matmul of window views

extension PoolGradientTests {
  /// pool over a transposed concrete (param) tensor: asStrided strides address the logical
  /// input, not the transposed memory (composeTransforms used to read the untransposed base).
  func testPoolOverTransposedParamForward() throws {
    LazyGraphContext.reset()
    let x = Tensor.param([4, 3], data: (0..<12).map(Float.init))
    let w = x.transpose([1, 0]).pool(kernel: [2], stride: [2])  // [3, 2, 2]
    XCTAssertEqual(try (w * 1.0).realize(), [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11])
  }

  func testPoolOverTransposeGrad() throws {
    let c = values(3 * 4 * 3, seed: 21)
    try check(leaves: [([9, 3], values(27, seed: 22))]) { p in
      let w = tanh(p[0]).transpose([1, 0]).pool(kernel: [3], stride: [2])  // [3, 4, 3]
      return (w * Tensor(c).reshape([3, 4, 3])).sum()
    }
  }

  /// matmul whose left operand is an im2col window view: the GEMM rewrite must not strip the
  /// view chain and read the source's raw memory.
  func testMatmulOfWindowView() throws {
    LazyGraphContext.reset()
    let x = Tensor((0..<10).map(Float.init)).reshape([2, 5])
    let cols = x.pool(kernel: [3], stride: [1]).transpose([1, 0, 2]).reshape([3, 6])
    let y = cols.matmul(Tensor([Float](repeating: 1, count: 6)).reshape([6, 1]))
    XCTAssertEqual(try y.realize(), [21, 27, 33])
  }
}
