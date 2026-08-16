import XCTest

@testable import DDSPE2E
@testable import DGenLazy

/// R2 piece 2: config/decoder wiring for the frequency-sampled noise branch.
/// In `fd` mode the noise head predicts one magnitude per FFT bin, so its width
/// is dictated by the FFT size rather than by `noiseFilterSize`.
final class NoiseFilterModeTests: XCTestCase {

  override func setUp() {
    super.setUp()
    LazyGraphContext.reset()
  }

  private func baseConfig() -> DDSPE2EConfig {
    var config = DDSPE2EConfig.default
    config.enableNoiseFilter = true
    config.modelHiddenSize = 16
    config.numHarmonics = 8
    return config
  }

  func testFDModeSizesNoiseHeadToBinCount() throws {
    var config = baseConfig()
    config.noiseFilterMode = .fd
    config.noiseFDFFTSize = 128
    config.noiseFilterSize = 15  // ignored in fd mode

    XCTAssertEqual(config.noiseFilterOutputSize, 65)

    let model = DDSPDecoderModel(config: config)
    XCTAssertEqual(model.W_filter?.shape.last, 65)
    XCTAssertEqual(model.b_filter?.shape.last, 65)
  }

  func testFIRModeKeepsTapCount() throws {
    var config = baseConfig()
    config.noiseFilterMode = .fir
    config.noiseFilterSize = 15

    XCTAssertEqual(config.noiseFilterOutputSize, 15)
    let model = DDSPDecoderModel(config: config)
    XCTAssertEqual(model.W_filter?.shape.last, 15)
  }

  func testFDValidationRejectsUnusableGeometry() throws {
    var nonPowerOfTwo = baseConfig()
    nonPowerOfTwo.noiseFilterMode = .fd
    nonPowerOfTwo.noiseFDFFTSize = 100
    XCTAssertThrowsError(try nonPowerOfTwo.validate())

    // An IR at least as long as the frame cannot be bounded by the window, which
    // is what keeps circular convolution from aliasing in time.
    var irTooLong = baseConfig()
    irTooLong.noiseFilterMode = .fd
    irTooLong.noiseFDFFTSize = 128
    irTooLong.noiseFDIRLength = 128
    XCTAssertThrowsError(try irTooLong.validate())

    var valid = baseConfig()
    valid.noiseFilterMode = .fd
    valid.noiseFDFFTSize = 128
    valid.noiseFDIRLength = 64
    XCTAssertNoThrow(try valid.validate())
  }
}
