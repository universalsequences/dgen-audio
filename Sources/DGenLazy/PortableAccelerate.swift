// PortableAccelerate - plain-Swift stand-ins for the small slice of vDSP that
// DGen's CPU-side analysis code uses.
//
// This file compiles to nothing on Apple platforms: there, the real Accelerate
// framework is used and behavior is unchanged. On Linux (and any other platform
// without Accelerate) it supplies source-compatible replacements so the
// spectral scorers / feature extractors keep building and produce the same
// numbers.
//
// Conventions deliberately match Accelerate:
//   * forward transforms use the exp(-2*pi*i*n*k/N) kernel
//   * vDSP_fft_zrip's packed real output stores DC in realp[0], Nyquist in
//     imagp[0], and every value is scaled by 2 relative to the plain DFT
//   * nothing is normalized by 1/N
//
// Only the operations actually called by this package are provided, and only
// with the strides those call sites use (all unit stride except vDSP_ctoz's
// interleaved input).

#if !canImport(Accelerate)

  import Foundation

  // MARK: - Types

  public typealias vDSP_Length = UInt
  public typealias vDSP_Stride = Int
  public typealias FFTRadix = Int32
  public typealias FFTDirection = Int32
  public typealias FFTSetup = OpaquePointer
  public typealias vDSP_DFT_Setup = OpaquePointer

  public let kFFTRadix2: Int = 0
  public let FFT_FORWARD: Int = 1
  public let FFT_INVERSE: Int = -1
  public let kFFTDirection_Forward: FFTDirection = 1
  public let kFFTDirection_Inverse: FFTDirection = -1

  public struct DSPComplex {
    public var real: Float
    public var imag: Float
    public init(real: Float = 0, imag: Float = 0) {
      self.real = real
      self.imag = imag
    }
  }

  public struct DSPSplitComplex {
    public var realp: UnsafeMutablePointer<Float>
    public var imagp: UnsafeMutablePointer<Float>
    public init(realp: UnsafeMutablePointer<Float>, imagp: UnsafeMutablePointer<Float>) {
      self.realp = realp
      self.imagp = imagp
    }
  }

  public enum vDSP_DFT_Direction {
    case FORWARD
    case INVERSE
  }

  // MARK: - Core radix-2 complex FFT

  /// In-place decimation-in-time radix-2 FFT over separate real/imaginary
  /// buffers of length `n` (a power of two). `inverse` conjugates the kernel and
  /// applies no 1/N scaling, matching vDSP.
  @inline(__always)
  private func portableFFT(
    _ real: UnsafeMutablePointer<Double>,
    _ imag: UnsafeMutablePointer<Double>,
    _ n: Int,
    inverse: Bool
  ) {
    guard n >= 2 else { return }

    // Bit-reversal permutation.
    var j = 0
    for i in 1..<n {
      var bit = n >> 1
      while j & bit != 0 {
        j ^= bit
        bit >>= 1
      }
      j ^= bit
      if i < j {
        real.swapAt(i, j)
        imag.swapAt(i, j)
      }
    }

    let sign: Double = inverse ? 1.0 : -1.0
    var twiddleReal = [Double](repeating: 0, count: n / 2)
    var twiddleImag = [Double](repeating: 0, count: n / 2)
    var len = 2
    while len <= n {
      let half = len >> 1
      let theta = sign * 2.0 * Double.pi / Double(len)
      for k in 0..<half {
        let angle = theta * Double(k)
        twiddleReal[k] = Foundation.cos(angle)
        twiddleImag[k] = Foundation.sin(angle)
      }
      for base in stride(from: 0, to: n, by: len) {
        for k in 0..<half {
          let wr = twiddleReal[k]
          let wi = twiddleImag[k]
          let a = base + k
          let b = a + half
          let br = real[b]
          let bi = imag[b]
          let tr = br * wr - bi * wi
          let ti = br * wi + bi * wr
          let ar = real[a]
          let ai = imag[a]
          real[a] = ar + tr
          imag[a] = ai + ti
          real[b] = ar - tr
          imag[b] = ai - ti
        }
      }
      len <<= 1
    }
  }

  extension UnsafeMutablePointer where Pointee == Double {
    @inline(__always)
    fileprivate func swapAt(_ i: Int, _ j: Int) {
      let tmp = self[i]
      self[i] = self[j]
      self[j] = tmp
    }
  }

  /// Naive O(n^2) DFT, used for the (unused in practice) non-power-of-two sizes
  /// that Accelerate's DFT API also accepts.
  private func portableDFT(
    inputReal: UnsafePointer<Float>,
    inputImag: UnsafePointer<Float>,
    outputReal: UnsafeMutablePointer<Float>,
    outputImag: UnsafeMutablePointer<Float>,
    n: Int,
    inverse: Bool
  ) {
    let sign: Double = inverse ? 1.0 : -1.0
    for k in 0..<n {
      var sumR = 0.0
      var sumI = 0.0
      for t in 0..<n {
        let angle = sign * 2.0 * Double.pi * Double(t) * Double(k) / Double(n)
        let c = Foundation.cos(angle)
        let s = Foundation.sin(angle)
        let xr = Double(inputReal[t])
        let xi = Double(inputImag[t])
        sumR += xr * c - xi * s
        sumI += xr * s + xi * c
      }
      outputReal[k] = Float(sumR)
      outputImag[k] = Float(sumI)
    }
  }

  // MARK: - vDSP_create_fftsetup / vDSP_fft_zrip

  private final class PortableFFTSetup {
    let log2n: vDSP_Length
    init(log2n: vDSP_Length) { self.log2n = log2n }
  }

  public func vDSP_create_fftsetup(_ log2n: vDSP_Length, _ radix: FFTRadix) -> FFTSetup? {
    let box = PortableFFTSetup(log2n: log2n)
    return OpaquePointer(Unmanaged.passRetained(box).toOpaque())
  }

  public func vDSP_destroy_fftsetup(_ setup: FFTSetup?) {
    guard let setup = setup else { return }
    Unmanaged<PortableFFTSetup>.fromOpaque(UnsafeRawPointer(setup)).release()
  }

  /// Interleaved -> split copy. `IC` is a stride in Float units over the
  /// interleaved input (2 for a packed complex vector); `IZ` strides the split
  /// output in complex elements.
  public func vDSP_ctoz(
    _ input: UnsafePointer<DSPComplex>,
    _ IC: vDSP_Stride,
    _ output: UnsafePointer<DSPSplitComplex>,
    _ IZ: vDSP_Stride,
    _ n: vDSP_Length
  ) {
    let floats = UnsafeRawPointer(input).assumingMemoryBound(to: Float.self)
    let split = output.pointee
    for i in 0..<Int(n) {
      split.realp[i * IZ] = floats[i * IC]
      split.imagp[i * IZ] = floats[i * IC + 1]
    }
  }

  /// Packed real forward/inverse FFT, matching vDSP_fft_zrip.
  ///
  /// Forward: input is a length-N real signal packed as realp[k] = x[2k],
  /// imagp[k] = x[2k+1] (N = 2^log2n, N/2 complex elements). Output is the
  /// packed half-spectrum, scaled by 2: realp[0] = 2*X[0], imagp[0] = 2*X[N/2],
  /// realp[k]/imagp[k] = 2*Re/Im(X[k]) for 0 < k < N/2.
  public func vDSP_fft_zrip(
    _ setup: FFTSetup,
    _ ioData: UnsafePointer<DSPSplitComplex>,
    _ stride: vDSP_Stride,
    _ log2n: vDSP_Length,
    _ direction: FFTDirection
  ) {
    precondition(stride == 1, "portable vDSP_fft_zrip supports unit stride only")
    let n = 1 << Int(log2n)
    guard n >= 2 else { return }
    let half = n / 2
    let split = ioData.pointee
    let inverse = direction != FFTDirection(FFT_FORWARD)

    var re = [Double](repeating: 0, count: n)
    var im = [Double](repeating: 0, count: n)

    if !inverse {
      // Rebuild the full real signal from its even/odd split packing.
      for k in 0..<half {
        re[2 * k] = Double(split.realp[k])
        re[2 * k + 1] = Double(split.imagp[k])
      }
      re.withUnsafeMutableBufferPointer { rp in
        im.withUnsafeMutableBufferPointer { ip in
          portableFFT(rp.baseAddress!, ip.baseAddress!, n, inverse: false)
        }
      }
      split.realp[0] = Float(2.0 * re[0])
      split.imagp[0] = Float(2.0 * re[half])
      for k in 1..<half {
        split.realp[k] = Float(2.0 * re[k])
        split.imagp[k] = Float(2.0 * im[k])
      }
    } else {
      // Packed half-spectrum -> full Hermitian spectrum, then inverse FFT.
      // vDSP applies no normalization here: a forward+inverse round trip comes
      // back scaled by 2*N (hence Apple's canonical 1/(2*N) rescale), which is
      // exactly the unnormalized inverse transform of the packed input.
      re[0] = Double(split.realp[0])
      im[0] = 0
      re[half] = Double(split.imagp[0])
      im[half] = 0
      for k in 1..<half {
        let xr = Double(split.realp[k])
        let xi = Double(split.imagp[k])
        re[k] = xr
        im[k] = xi
        re[n - k] = xr
        im[n - k] = -xi
      }
      re.withUnsafeMutableBufferPointer { rp in
        im.withUnsafeMutableBufferPointer { ip in
          portableFFT(rp.baseAddress!, ip.baseAddress!, n, inverse: true)
        }
      }
      for k in 0..<half {
        split.realp[k] = Float(re[2 * k])
        split.imagp[k] = Float(re[2 * k + 1])
      }
    }
  }

  /// Split-complex in-place FFT, matching vDSP_fft_zip (no 1/N on the inverse).
  public func vDSP_fft_zip(
    _ setup: FFTSetup,
    _ ioData: UnsafePointer<DSPSplitComplex>,
    _ stride: vDSP_Stride,
    _ log2n: vDSP_Length,
    _ direction: FFTDirection
  ) {
    precondition(stride == 1, "portable vDSP_fft_zip supports unit stride only")
    let n = 1 << Int(log2n)
    guard n >= 2 else { return }
    let split = ioData.pointee
    let inverse = direction != FFTDirection(FFT_FORWARD)
    var re = [Double](repeating: 0, count: n)
    var im = [Double](repeating: 0, count: n)
    for i in 0..<n {
      re[i] = Double(split.realp[i])
      im[i] = Double(split.imagp[i])
    }
    re.withUnsafeMutableBufferPointer { rp in
      im.withUnsafeMutableBufferPointer { ip in
        portableFFT(rp.baseAddress!, ip.baseAddress!, n, inverse: inverse)
      }
    }
    for i in 0..<n {
      split.realp[i] = Float(re[i])
      split.imagp[i] = Float(im[i])
    }
  }

  // MARK: - vDSP DFT API

  private final class PortableDFTSetup {
    let n: Int
    let inverse: Bool
    init(n: Int, inverse: Bool) {
      self.n = n
      self.inverse = inverse
    }
  }

  public func vDSP_DFT_zop_CreateSetup(
    _ previous: vDSP_DFT_Setup?,
    _ length: vDSP_Length,
    _ direction: vDSP_DFT_Direction
  ) -> vDSP_DFT_Setup? {
    let n = Int(length)
    guard n > 0 else { return nil }
    let box = PortableDFTSetup(n: n, inverse: direction == .INVERSE)
    return OpaquePointer(Unmanaged.passRetained(box).toOpaque())
  }

  public func vDSP_DFT_DestroySetup(_ setup: vDSP_DFT_Setup?) {
    guard let setup = setup else { return }
    Unmanaged<PortableDFTSetup>.fromOpaque(UnsafeRawPointer(setup)).release()
  }

  public func vDSP_DFT_Execute(
    _ setup: vDSP_DFT_Setup,
    _ inputReal: UnsafePointer<Float>,
    _ inputImaginary: UnsafePointer<Float>,
    _ outputReal: UnsafeMutablePointer<Float>,
    _ outputImaginary: UnsafeMutablePointer<Float>
  ) {
    let box = Unmanaged<PortableDFTSetup>.fromOpaque(UnsafeRawPointer(setup))
      .takeUnretainedValue()
    let n = box.n
    guard n > 0 else { return }

    if n & (n - 1) == 0 {
      var re = [Double](repeating: 0, count: n)
      var im = [Double](repeating: 0, count: n)
      for i in 0..<n {
        re[i] = Double(inputReal[i])
        im[i] = Double(inputImaginary[i])
      }
      re.withUnsafeMutableBufferPointer { rp in
        im.withUnsafeMutableBufferPointer { ip in
          portableFFT(rp.baseAddress!, ip.baseAddress!, n, inverse: box.inverse)
        }
      }
      for i in 0..<n {
        outputReal[i] = Float(re[i])
        outputImaginary[i] = Float(im[i])
      }
    } else {
      portableDFT(
        inputReal: inputReal, inputImag: inputImaginary,
        outputReal: outputReal, outputImag: outputImaginary,
        n: n, inverse: box.inverse)
    }
  }

  // MARK: - Elementwise / reduction helpers

  public func vDSP_vmul(
    _ a: UnsafePointer<Float>,
    _ ia: vDSP_Stride,
    _ b: UnsafePointer<Float>,
    _ ib: vDSP_Stride,
    _ c: UnsafeMutablePointer<Float>,
    _ ic: vDSP_Stride,
    _ n: vDSP_Length
  ) {
    for i in 0..<Int(n) {
      c[i * ic] = a[i * ia] * b[i * ib]
    }
  }

  public func vDSP_dotpr(
    _ a: UnsafePointer<Float>,
    _ ia: vDSP_Stride,
    _ b: UnsafePointer<Float>,
    _ ib: vDSP_Stride,
    _ result: UnsafeMutablePointer<Float>,
    _ n: vDSP_Length
  ) {
    var sum: Float = 0
    for i in 0..<Int(n) {
      sum += a[i * ia] * b[i * ib]
    }
    result.pointee = sum
  }

#endif  // !canImport(Accelerate)
