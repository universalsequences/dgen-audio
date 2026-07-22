import XCTest

@testable import DGen
@testable import DGenLazy

/// Cumulative (prefix) sum along an axis. Inference-only op (no backward yet).
/// The efficient O(N) building block for wide spectral smoothing on the C RT path
/// (box filter = `cumsum[i+w] - cumsum[i-w]`), avoiding O(N·W) conv or O(N²) matmul.
final class CumsumTests: XCTestCase {

  override func setUp() {
    super.setUp()
    LazyGraphContext.reset()
  }

  private func cumsum1D(backend: Backend, _ data: [Float]) throws -> [Float] {
    DGenConfig.backend = backend
    LazyGraphContext.reset()
    return try Tensor(data).cumsum().realize()
  }

  func test1D_BothBackends() throws {
    for backend in [Backend.c, .metal] {
      let r = try cumsum1D(backend: backend, [1, 2, 3, 4, 5])
      let name = backend == .c ? "C" : "Metal"
      XCTAssertEqual(r, [1, 3, 6, 10, 15], "[\(name)] 1D cumsum")
    }
  }

  func test1D_NegativesAndZeros() throws {
    for backend in [Backend.c, .metal] {
      let r = try cumsum1D(backend: backend, [2, -1, 0, 3, -4])
      XCTAssertEqual(r, [2, 1, 1, 4, 0])
    }
  }

  /// 2D cumsum along the last axis (default): each row scanned independently.
  func test2D_LastAxis() throws {
    for backend in [Backend.c, .metal] {
      DGenConfig.backend = backend
      LazyGraphContext.reset()
      let t = Tensor([[1, 2, 3], [4, 5, 6]])  // [2,3]
      let r = try t.cumsum(axis: -1).realize()
      // Row-major: [1,3,6, 4,9,15]
      XCTAssertEqual(r, [1, 3, 6, 4, 9, 15], "[\(backend)] axis=-1")
    }
  }

  /// 2D cumsum along axis 0 (strided scan: inner stride = numCols).
  func test2D_Axis0() throws {
    for backend in [Backend.c, .metal] {
      DGenConfig.backend = backend
      LazyGraphContext.reset()
      let t = Tensor([[1, 2, 3], [4, 5, 6]])  // [2,3]
      let r = try t.cumsum(axis: 0).realize()
      // Column-wise running sum: [1,2,3, 5,7,9]
      XCTAssertEqual(r, [1, 2, 3, 5, 7, 9], "[\(backend)] axis=0")
    }
  }

  /// Box-filter (moving average) via cumsum difference — the soothe smoothing use.
  func testBoxFilterViaCumsumDifference() throws {
    // smooth[i] = (prefix[i+w] - prefix[i-w-1]) / window, here done by hand to
    // confirm cumsum gives a usable prefix array.
    let r = try cumsum1D(backend: .c, [1, 1, 1, 1, 1, 1, 1, 1])
    XCTAssertEqual(r, [1, 2, 3, 4, 5, 6, 7, 8])
    // window of 3 centered at i=4: prefix[5] - prefix[2] = 6 - 3 = 3 → mean 1.0
    XCTAssertEqual((r[5] - r[2]) / 3, 1.0, accuracy: 1e-5)
  }

  /// Per-frame signalTensor path: cumsum of a hop-gated buffered window.
  /// Exercises frame-aware tensor addressing and confirms C == Metal.
  func testSignalTensorCumsum_BothBackends() throws {
    let size = 4
    func run(_ backend: Backend) throws -> [Float] {
      DGenConfig.backend = backend
      LazyGraphContext.reset()
      // Constant 1.0 → buffered [size] window (per-frame). cumsum → ramp.
      let sig = Signal.constant(1.0)
      let win = sig.buffer(size: size, hop: 1).reshape([size])
      // Sum of the cumsum: once the ring fills, window is [1,1,1,1] →
      // cumsum [1,2,3,4] → sum 10.
      return try win.cumsum().sum().realize(frames: 8)
    }
    let c = try run(.c)
    let metal = try run(.metal)
    XCTAssertEqual(c, metal, "C and Metal cumsum must agree")
    // After the ring fills (frame >= size-1) the sum is 1+2+3+4 = 10.
    for i in (size - 1)..<c.count {
      XCTAssertEqual(c[i], 10.0, accuracy: 1e-4, "frame \(i)")
    }
  }
}
