import Foundation
import XCTest

@testable import DGenLazy

final class TransformerOpsTests: XCTestCase {

  override func setUp() {
    super.setUp()
    LazyGraphContext.reset()
  }

  // MARK: - maxAxis

  func testMaxAxisLastDim() throws {
    // [2, 3] tensor, max along axis 1 → [2]
    let t = Tensor([[1, 5, 3], [4, 2, 6]])
    let m = t.max(axis: 1)
    XCTAssertEqual(m.shape, [2])

    let result = try m.realize()
    XCTAssertEqual(result[0], 5.0, accuracy: 1e-5, "max of row [1,5,3] = 5")
    XCTAssertEqual(result[1], 6.0, accuracy: 1e-5, "max of row [4,2,6] = 6")
  }

  func testMaxAxisFirstDim() throws {
    // [2, 3] tensor, max along axis 0 → [3]
    let t = Tensor([[1, 5, 3], [4, 2, 6]])
    let m = t.max(axis: 0)
    XCTAssertEqual(m.shape, [3])

    let result = try m.realize()
    XCTAssertEqual(result[0], 4.0, accuracy: 1e-5)
    XCTAssertEqual(result[1], 5.0, accuracy: 1e-5)
    XCTAssertEqual(result[2], 6.0, accuracy: 1e-5)
  }

  func testMaxAxisNegative() throws {
    // Negative axis: -1 means last dim
    let t = Tensor([[10, 20], [30, 5]])
    let m = t.max(axis: -1)
    XCTAssertEqual(m.shape, [2])

    let result = try m.realize()
    XCTAssertEqual(result[0], 20.0, accuracy: 1e-5)
    XCTAssertEqual(result[1], 30.0, accuracy: 1e-5)
  }

  // MARK: - meanAxis

  func testMeanAxisLastDim() throws {
    // [2, 3] tensor, mean along axis 1 → [2]
    let t = Tensor([[3, 6, 9], [10, 20, 30]])
    let m = t.mean(axis: 1)
    XCTAssertEqual(m.shape, [2])

    let result = try m.realize()
    XCTAssertEqual(result[0], 6.0, accuracy: 1e-5, "mean of [3,6,9] = 6")
    XCTAssertEqual(result[1], 20.0, accuracy: 1e-5, "mean of [10,20,30] = 20")
  }

  func testMeanAxisFirstDim() throws {
    // [2, 3] tensor, mean along axis 0 → [3]
    let t = Tensor([[2, 4, 6], [8, 10, 12]])
    let m = t.mean(axis: 0)
    XCTAssertEqual(m.shape, [3])

    let result = try m.realize()
    XCTAssertEqual(result[0], 5.0, accuracy: 1e-5)
    XCTAssertEqual(result[1], 7.0, accuracy: 1e-5)
    XCTAssertEqual(result[2], 9.0, accuracy: 1e-5)
  }

  // MARK: - softmax

  func testSoftmaxRowsSumToOne() throws {
    // softmax along last axis: each row should sum to 1
    let t = Tensor([[1, 2, 3], [1, 1, 1]])
    let s = t.softmax(axis: -1)
    XCTAssertEqual(s.shape, [2, 3])

    let result = try s.realize()
    // Row 0 sums to 1
    let row0Sum = result[0] + result[1] + result[2]
    XCTAssertEqual(row0Sum, 1.0, accuracy: 1e-4, "softmax row should sum to 1")
    // Row 1 sums to 1
    let row1Sum = result[3] + result[4] + result[5]
    XCTAssertEqual(row1Sum, 1.0, accuracy: 1e-4, "softmax row should sum to 1")
  }

  func testSoftmaxValuesInRange() throws {
    let t = Tensor([[1, 2, 3], [10, -10, 0]])
    let s = t.softmax(axis: -1)
    let result = try s.realize()

    for val in result {
      XCTAssertGreaterThanOrEqual(val, 0.0, "softmax values >= 0")
      XCTAssertLessThanOrEqual(val, 1.0, "softmax values <= 1")
    }
  }

  func testSoftmaxUniformInput() throws {
    // Equal inputs → uniform distribution
    let t = Tensor([[5, 5, 5]])
    let s = t.softmax(axis: -1)
    let result = try s.realize()

    for val in result {
      XCTAssertEqual(val, 1.0 / 3.0, accuracy: 1e-4, "uniform input → uniform softmax")
    }
  }

  func testSoftmaxNumericalStability() throws {
    // Large values shouldn't cause overflow (max subtraction handles this)
    let t = Tensor([[1000, 1001, 1002]])
    let s = t.softmax(axis: -1)
    let result = try s.realize()

    let sum = result.reduce(0, +)
    XCTAssertEqual(sum, 1.0, accuracy: 1e-4, "softmax of large values should still sum to 1")
    for val in result {
      XCTAssertFalse(val.isNaN, "softmax should not produce NaN")
      XCTAssertFalse(val.isInfinite, "softmax should not produce Inf")
    }
  }

  // MARK: - softmax backward

  func testSoftmaxBackward() throws {
    let learnable = Tensor([[1, 5, 4]], requiresGrad: true)
    let soft = learnable.softmax(axis: -1)
    let target = Tensor([[0.2, 0.5, 0.6]])
    let diff = (soft - target)
    let mse = (diff * diff).mean()
    _ = try mse.backward(frameCount: 1)
    XCTAssertNotNil(learnable.grad, "grad should not be nil")
    if let gradA = learnable.grad {
      let gradValues = try gradA.realize()
      let anyNonZero = gradValues.contains { $0 != 0.0 }
      XCTAssertTrue(anyNonZero, "grad must have at least one non-zero element")
    }
  }

  // MARK: - Scaled Dot-Product Attention

  /// Compute scaled dot-product attention from primitives.
  ///
  /// - Parameters:
  ///   - X: input [seq_len, d_model]
  ///   - WQ: query projection [d_model, d_k]
  ///   - WK: key projection [d_model, d_k]
  ///   - WV: value projection [d_model, d_v]
  ///   - mask: optional [seq_len, seq_len] mask added to scores before softmax (use -1e9 to block)
  /// - Returns: (output [seq_len, d_v], attn_weights [seq_len, seq_len])
  static func attention(
    _ X: Tensor, _ WQ: Tensor, _ WK: Tensor, _ WV: Tensor, mask: Tensor? = nil
  )
    -> (output: Tensor, weights: Tensor)
  {
    let dk = WQ.shape[1]
    let Q = X.matmul(WQ)
    let K = X.matmul(WK)
    let V = X.matmul(WV)
    var scores = Q.matmul(K.transpose()) / Foundation.sqrt(Float(dk))
    if let mask = mask {
      scores = scores + mask
    }
    let weights = scores.softmax(axis: -1)
    return (output: weights.matmul(V), weights: weights)
  }

  func testAttentionForward() throws {
    // X: [4, 3] - 4 sequence positions, 3-dim embedding
    let X = Tensor([
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
      [1, 1, 0],
    ])

    // Identity-ish projections: W_Q = W_K = W_V = I
    let I3 = Tensor([
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ])

    let (output, weights) = Self.attention(X, I3, I3, I3)

    // Verify shapes
    XCTAssertEqual(output.shape, [4, 3], "Output should be [seq_len, d_v]")
    XCTAssertEqual(weights.shape, [4, 4], "Weights should be [seq_len, seq_len]")

    let weightVals = try weights.realize()
    let outputVals = try output.realize()

    // Attention weights: all in [0, 1]
    for val in weightVals {
      XCTAssertGreaterThanOrEqual(val, 0.0, "weight >= 0")
      XCTAssertLessThanOrEqual(val, 1.0, "weight <= 1")
      XCTAssertFalse(val.isNaN, "no NaN in weights")
    }

    // Each row of weights sums to ~1
    let seqLen = 4
    for row in 0..<seqLen {
      let rowSum = (0..<seqLen).map { weightVals[row * seqLen + $0] }.reduce(0, +)
      XCTAssertEqual(rowSum, 1.0, accuracy: 1e-4, "attention weight row \(row) sums to 1")
    }

    // Output: no NaN/Inf
    for val in outputVals {
      XCTAssertFalse(val.isNaN, "no NaN in output")
      XCTAssertFalse(val.isInfinite, "no Inf in output")
    }
  }

  func testAttentionLearning() throws {
    // Teacher: fixed projection matrices
    let teacherWQ = Tensor([
      [0.5, 0.1, -0.3],
      [-0.2, 0.8, 0.1],
      [0.3, -0.1, 0.6],
    ])
    let teacherWK = Tensor([
      [0.4, -0.2, 0.5],
      [0.1, 0.7, -0.1],
      [-0.3, 0.2, 0.4],
    ])
    let teacherWV = Tensor([
      [0.6, 0.0, -0.2],
      [-0.1, 0.5, 0.3],
      [0.2, -0.3, 0.7],
    ])

    // Student: learnable params initialized to small values
    let studentWQ = Tensor.param(
      [3, 3],
      data: [
        0.1, 0.1, 0.1,
        0.1, 0.1, 0.1,
        0.1, 0.1, 0.1,
      ])
    let studentWK = Tensor.param(
      [3, 3],
      data: [
        0.1, 0.1, 0.1,
        0.1, 0.1, 0.1,
        0.1, 0.1, 0.1,
      ])
    let studentWV = Tensor.param(
      [3, 3],
      data: [
        0.1, 0.1, 0.1,
        0.1, 0.1, 0.1,
        0.1, 0.1, 0.1,
      ])

    // Input X: [4, 3]
    let X = Tensor([
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
      [1, 1, 0],
    ])

    let optimizer = Adam(params: [studentWQ, studentWK, studentWV], lr: 0.01)
    let epochs = 50
    var losses: [Float] = []

    for epoch in 0..<epochs {
      // Rebuild graph fresh each iteration
      let (teacherOut, _) = Self.attention(X, teacherWQ, teacherWK, teacherWV)
      let (studentOut, _) = Self.attention(X, studentWQ, studentWK, studentWV)

      let diff = studentOut - teacherOut
      let loss = (diff * diff).sum()

      let lossValue = try loss.backward(frameCount: 1).first ?? 0
      losses.append(lossValue)

      if epoch % 10 == 0 {
        print("Attention epoch \(epoch): loss=\(lossValue)")
      }

      optimizer.step()
      optimizer.zeroGrad()
    }

    print("Attention training: initial=\(losses[0]) final=\(losses.last!)")

    // Loss should decrease significantly
    XCTAssertGreaterThan(losses[0], losses.last!, "Loss should decrease")
    XCTAssertLessThan(losses.last!, losses[0] * 0.1, "Final loss < 10% of initial")
  }

  // MARK: - Key-Value Retrieval

  func testAttentionKeyValueRetrieval() throws {
    // Input [4, 4]: dims 0-1 are "keys", dims 2-3 are "values"
    // Group A (rows 0, 2): key=[3,0]  Group B (rows 1, 3): key=[0,3]
    let X = Tensor([
      [3, 0, 1, 2],  // group A, value=[1, 2]
      [0, 3, 3, 4],  // group B, value=[3, 4]
      [3, 0, 5, 6],  // group A, value=[5, 6]
      [0, 3, 7, 8],  // group B, value=[7, 8]
    ])

    // Teacher: structured projections that extract keys (dims 0-1) and values (dims 2-3)
    // WQ/WK select dims 0-1 as "key space", WV selects dims 2-3 as "value space"
    let teacherWQ = Tensor([[1, 0], [0, 1], [0, 0], [0, 0]])  // [4, 2]
    let teacherWK = Tensor([[1, 0], [0, 1], [0, 0], [0, 0]])
    let teacherWV = Tensor([[0, 0], [0, 0], [1, 0], [0, 1]])

    // Student: uniform init (deterministic), must discover the key-value decomposition
    let init4x2: [Float] = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    let studentWQ = Tensor.param([4, 2], data: init4x2)
    let studentWK = Tensor.param([4, 2], data: init4x2)
    let studentWV = Tensor.param([4, 2], data: init4x2)

    let optimizer = Adam(params: [studentWQ, studentWK, studentWV], lr: 0.01)
    var losses: [Float] = []

    for epoch in 0..<200 {
      let (teacherOut, _) = Self.attention(X, teacherWQ, teacherWK, teacherWV)
      let (studentOut, _) = Self.attention(X, studentWQ, studentWK, studentWV)
      let diff = studentOut - teacherOut
      let loss = (diff * diff).sum()

      let lossValue = try loss.backward(frameCount: 1).first ?? 0
      losses.append(lossValue)

      if epoch % 40 == 0 {
        print("KV retrieval epoch \(epoch): loss=\(lossValue)")
      }

      optimizer.step()
      optimizer.zeroGrad()
    }

    print("KV retrieval: initial=\(losses[0]) final=\(losses.last!)")
    XCTAssertLessThan(losses.last!, losses[0] * 0.1, "Loss should decrease 10x")

    // Verify the student actually performs key-value retrieval
    // Teacher output: group A rows get avg([1,2],[5,6])=[3,4], group B gets avg([3,4],[7,8])=[5,6]
    let (studentOut, _) = Self.attention(X, studentWQ, studentWK, studentWV)
    let output = try studentOut.realize()

    // Output is [4, 2] flattened: [row0_d0, row0_d1, row1_d0, row1_d1, ...]
    XCTAssertEqual(output[0], 3.0, accuracy: 1.5, "Row 0 (group A) retrieves ~[3, _]")
    XCTAssertEqual(output[1], 4.0, accuracy: 1.5, "Row 0 (group A) retrieves ~[_, 4]")
    XCTAssertEqual(output[2], 5.0, accuracy: 1.5, "Row 1 (group B) retrieves ~[5, _]")
    XCTAssertEqual(output[3], 6.0, accuracy: 1.5, "Row 1 (group B) retrieves ~[_, 6]")
  }

  // MARK: - Next Token Prediction

  func testNextTokenPrediction() throws {
    // Vocabulary: 4 tokens as one-hot vectors
    // A=[1,0,0,0], B=[0,1,0,0], C=[0,0,1,0], D=[0,0,0,1]
    // Cyclic pattern: A→B→C→D→A

    // Input sequence: [A, B, C, D]
    let X = Tensor([
      [1, 0, 0, 0],  // A
      [0, 1, 0, 0],  // B
      [0, 0, 1, 0],  // C
      [0, 0, 0, 1],  // D
    ])

    // Target: next token for each position
    let target = Tensor([
      [0, 1, 0, 0],  // A → B
      [0, 0, 1, 0],  // B → C
      [0, 0, 0, 1],  // C → D
      [1, 0, 0, 0],  // D → A
    ])

    // Causal mask: position i can only attend to positions 0..i
    // -1e9 in blocked positions → ~0 after softmax
    let causalMask = Tensor([
      [0, -1e9, -1e9, -1e9],
      [0, 0, -1e9, -1e9],
      [0, 0, 0, -1e9],
      [0, 0, 0, 0],
    ])

    // Learnable parameters — non-uniform init to break symmetry
    let WQ = Tensor.param(
      [4, 4],
      data: [
        0.20, -0.10, 0.05, 0.15, -0.05, 0.20, 0.10, -0.15,
        0.10, 0.05, -0.20, 0.15, -0.15, 0.10, 0.05, 0.20,
      ])
    let WK = Tensor.param(
      [4, 4],
      data: [
        0.15, 0.05, -0.10, 0.20, 0.10, -0.15, 0.20, 0.05,
        -0.20, 0.15, 0.05, -0.10, 0.05, 0.20, -0.15, 0.10,
      ])
    let WV = Tensor.param(
      [4, 4],
      data: [
        -0.10, 0.20, 0.15, 0.05, 0.15, -0.05, 0.20, -0.10,
        0.05, 0.10, -0.15, 0.20, 0.20, -0.10, 0.05, 0.15,
      ])
    let WO = Tensor.param(
      [4, 4],
      data: [
        0.05, 0.15, -0.20, 0.10, -0.10, 0.05, 0.15, 0.20,
        0.20, -0.15, 0.10, 0.05, 0.10, 0.20, 0.05, -0.15,
      ])

    let optimizer = Adam(params: [WQ, WK, WV, WO], lr: 0.01)
    var losses: [Float] = []

    for epoch in 0..<200 {
      // Causal attention + output projection
      let (attnOut, _) = Self.attention(X, WQ, WK, WV, mask: causalMask)
      let logits = attnOut.matmul(WO)  // [4, 4] — one "logit" per vocab token per position

      let diff = logits - target
      let loss = (diff * diff).sum()

      let lossValue = try loss.backward(frameCount: 1).first ?? 0
      losses.append(lossValue)

      if epoch % 40 == 0 {
        print("Next-token epoch \(epoch): loss=\(lossValue)")
      }

      optimizer.step()
      optimizer.zeroGrad()
    }

    print("Next-token: initial=\(losses[0]) final=\(losses.last!)")
    XCTAssertLessThan(losses.last!, losses[0] * 0.1, "Loss should decrease 10x")

    // Verify: for each position, the highest logit should be the correct next token
    let (attnOut, _) = Self.attention(X, WQ, WK, WV, mask: causalMask)
    let logits = attnOut.matmul(WO)
    let output = try logits.realize()

    // Check each position predicts the right next token (argmax)
    let expectedArgmax = [1, 2, 3, 0]  // B, C, D, A
    for pos in 0..<4 {
      let row = (0..<4).map { output[pos * 4 + $0] }
      let predicted = row.enumerated().max(by: { $0.element < $1.element })!.offset
      print(
        "Position \(pos): logits=\(row.map { String(format: "%.2f", $0) }) predicted=\(predicted) expected=\(expectedArgmax[pos])"
      )
      XCTAssertEqual(
        predicted, expectedArgmax[pos],
        "Position \(pos) should predict token \(expectedArgmax[pos])")
    }
  }

  // MARK: - Rhythm Onset Detection

  /// N-point FFT using tensor view ops (reshape, transpose, shrink, pad, expand).
  /// Copied from TensorFFTTests — Cooley-Tukey butterfly decomposed into tensor ops.
  private func tensorFFT(_ input: Tensor, N: Int) -> (re: Tensor, im: Tensor) {
    let k = Int(Foundation.log2(Double(N)))
    precondition(1 << k == N, "N must be a power of 2")

    // Bit-reversal: reshape to [2,2,...,2], reverse axes, flatten
    let twos = [Int](repeating: 2, count: k)
    var re = input.reshape(twos)
      .transpose(Array((0..<k).reversed()))
      .reshape([N])
    var im = Tensor.zeros([N])

    // k butterfly stages
    for s in 0..<k {
      let half = 1 << s
      let blocks = N / (2 * half)

      let re3d = re.reshape([blocks, 2, half])
      let im3d = im.reshape([blocks, 2, half])

      let even_re = re3d.shrink([nil, (0, 1), nil]).reshape([blocks, half])
      let odd_re = re3d.shrink([nil, (1, 2), nil]).reshape([blocks, half])
      let even_im = im3d.shrink([nil, (0, 1), nil]).reshape([blocks, half])
      let odd_im = im3d.shrink([nil, (1, 2), nil]).reshape([blocks, half])

      // Twiddle factors (precomputed on CPU)
      var twRe = [Float](repeating: 0, count: half)
      var twIm = [Float](repeating: 0, count: half)
      for j in 0..<half {
        let angle = -2.0 * Float.pi * Float(j) / Float(2 * half)
        twRe[j] = Foundation.cos(angle)
        twIm[j] = Foundation.sin(angle)
      }

      let twiddleRe = Tensor(twRe).reshape([1, half]).expand([blocks, half])
      let twiddleIm = Tensor(twIm).reshape([1, half]).expand([blocks, half])

      let t_re = odd_re * twiddleRe - odd_im * twiddleIm
      let t_im = odd_re * twiddleIm + odd_im * twiddleRe

      let top_re = even_re + t_re
      let top_im = even_im + t_im
      let bot_re = even_re - t_re
      let bot_im = even_im - t_im

      re = (top_re.pad([(0, 0), (0, half)]) + bot_re.pad([(0, 0), (half, 0)])).reshape([N])
      im = (top_im.pad([(0, 0), (0, half)]) + bot_im.pad([(0, 0), (half, 0)])).reshape([N])
    }

    return (re, im)
  }

  func testRhythmOnsetDetection() throws {
    // === Step 1: Generate synthetic drum audio ===
    // 4 time windows × 8 samples each = 32 total samples
    // Pattern: alternating hit/rest [1, 0, 1, 0]
    let windowSize = 8
    let numWindows = 4
    let numBins = windowSize / 2  // 4 positive-frequency bins
    let pattern: [Float] = [1, 0, 1, 0]

    // Generate kick drum: decaying sine, slightly different freq per hit
    var audio = [Float](repeating: 0, count: windowSize * numWindows)
    for h in 0..<numWindows {
      if pattern[h] > 0.5 {
        let freq: Float = 2.0 + Float(h) * 0.5
        for i in 0..<windowSize {
          let t = Float(i) / Float(windowSize)
          let decay = Foundation.pow(1.0 - t, 2)
          audio[h * windowSize + i] = sin(t * 2.0 * Float.pi * freq) * decay
        }
      }
    }

    // === Step 2: Compute spectrogram using tensor FFT ===
    var spectrogramData = [Float]()

    for w in 0..<numWindows {
      LazyGraphContext.reset()
      let windowData = Array(audio[w * windowSize..<(w + 1) * windowSize])
      let windowTensor = Tensor(windowData)
      let (re, im) = tensorFFT(windowTensor, N: windowSize)
      let magSq = re * re + im * im
      let magValues = try magSq.realize()
      spectrogramData.append(contentsOf: Array(magValues[0..<numBins]))
    }

    // Reset for the training graph
    LazyGraphContext.reset()

    // Non-uniform init for symmetry breaking
    let WQ = Tensor.param(
      [4, 4],
      data: [
        0.20, -0.10, 0.05, 0.15, -0.05, 0.20, 0.10, -0.15,
        0.10, 0.05, -0.20, 0.15, -0.15, 0.10, 0.05, 0.20,
      ])
    let WK = Tensor.param(
      [4, 4],
      data: [
        0.15, 0.05, -0.10, 0.20, 0.10, -0.15, 0.20, 0.05,
        -0.20, 0.15, 0.05, -0.10, 0.05, 0.20, -0.15, 0.10,
      ])
    let WV = Tensor.param(
      [4, 4],
      data: [
        -0.10, 0.20, 0.15, 0.05, 0.15, -0.05, 0.20, -0.10,
        0.05, 0.10, -0.15, 0.20, 0.20, -0.10, 0.05, 0.15,
      ])
    let WO = Tensor.param([4, 1], data: [0.20, -0.15, 0.10, -0.05])

    let optimizer = Adam(params: [WQ, WK, WV, WO], lr: 0.05)
    var losses: [Float] = []

    for epoch in 0..<70 {
      // Spectrogram: [4, 4] — 4 time windows × 4 frequency bins
      let spectrogram = Tensor(spectrogramData).reshape([numWindows, numBins])

      // === Step 3: Train attention to detect onsets ===
      let target = Tensor(pattern).reshape([numWindows, 1])

      let (attnOut, _) = Self.attention(spectrogram, WQ, WK, WV)
      let predictions = attnOut.matmul(WO)  // [4, 1] raw scores (no sigmoid)

      let diff = predictions - target
      let loss = (diff * diff).sum()

      if epoch == 0 {
        DGenConfig.kernelOutputPath = "/tmp/attention_transcription.metal"
        DGenConfig.debug = true
      }
      let lossValue = try loss.backward(frameCount: 1).first ?? 0
      DGenConfig.kernelOutputPath = nil
      DGenConfig.debug = false
      losses.append(lossValue)

      if epoch % 10 == 0 {
        print("Rhythm epoch \(epoch): loss=\(lossValue)")
      }

      optimizer.step()
      optimizer.zeroGrad()
    }

    print("Rhythm onset: initial=\(losses[0]) final=\(losses.last!)")
    XCTAssertLessThan(losses.last!, losses[0] * 0.5, "Loss should decrease 2x")

    let spectrogram = Tensor(spectrogramData).reshape([numWindows, numBins])

    // === Step 4: Verify onset detection ===
    let (finalAttn, _) = Self.attention(spectrogram, WQ, WK, WV)
    let finalPred = finalAttn.matmul(WO)
    let predValues = try finalPred.realize()

    print("Rhythm predictions vs pattern:")
    for w in 0..<numWindows {
      let label = pattern[w] > 0.5 ? "HIT" : "rest"
      print("  Window \(w) [\(label)]: \(String(format: "%.3f", predValues[w]))")
    }

    // Hit predictions should be higher than rest predictions
    let hitPreds = (0..<numWindows).filter { pattern[$0] > 0.5 }.map { predValues[$0] }
    let restPreds = (0..<numWindows).filter { pattern[$0] < 0.5 }.map { predValues[$0] }
    let minHit = hitPreds.min()!
    let maxRest = restPreds.max()!
    XCTAssertGreaterThan(minHit, maxRest, "All hits should score higher than all rests")
  }
}
