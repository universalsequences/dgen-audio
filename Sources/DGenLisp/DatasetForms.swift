// DatasetForms.swift — the dataset-training forms.
//
// A patch written over ROW tensors ([B, w]: one row per kick, B = 1 in the instrument) can be
// trained by `dgenlisp train-dataset` against a folder of targets, then compiled unchanged.
// Every form here is identity-like in a compile; a `DatasetContext` attached by the trainer
// swaps in learnable leaves:
//
//   (latent-table @name codes @dim D X)            compile: X ([1,D]); train: [B,D] code rows
//   (tensor @shape [r c] @file "w.json" @learnable true [@init glorot|zeros|small])
//                                                   compile: the file; train: a learnable leaf
//   (row-layout NAME (SEG width squash [lo hi]) …)  registers a decoder-row layout
//   (seg U NAME SEG)                                squashed natural slice [B, width] of U
//   (param-row U @layout NAME)                      decoder/voice split point (see DatasetContext)
//   (row-out Y)                                     compile: Y's single element as a signal;
//                                                   train: the [B] lane signal the losses read
//   (hit-time TRIG [@max S])                        seconds since TRIG's last rising edge
//                                                   (S before the first one); train: a
//                                                   frame-parallel ramp from frame 0
//
// Under a dataset context every hit is one trigger at frame 0, so `latch`, `event-hold` and
// `event-latch` of a static value lower to the value itself (AD rejects event-hold, and latch
// only passes gradient on trigger frames). A frame-varying latched value is an error: that
// lowering would be a silently wrong answer.

import DGen
import DGenLazy
import Foundation

/// Squash from an unbounded decoder output u to a natural value.
enum RowSquash: String, Codable {
  /// lo · (hi/lo)^σ(u)
  case log
  /// lo + (hi − lo) σ(u)
  case lin
  /// u itself
  case raw
}

struct RowSegment: Codable, Equatable {
  let name: String
  let offset: Int
  let width: Int
  let squash: RowSquash
  let lo: Float
  let hi: Float

  /// Natural value → u (the inverse squash), clamped `margin` inside the box so analysis targets
  /// at the edges stay finite.
  func toU(_ natural: Float, margin: Float = 1e-4) -> Float {
    func logit(_ s: Float) -> Float {
      let c = Swift.min(Swift.max(s, margin), 1 - margin)
      return Foundation.log(c / (1 - c))
    }
    switch squash {
    case .raw: return natural
    case .lin: return logit((natural - lo) / (hi - lo))
    case .log:
      let v = Swift.max(natural, Float.leastNormalMagnitude)
      return logit((Foundation.log(v) - Foundation.log(lo)) / (Foundation.log(hi) - Foundation.log(lo)))
    }
  }

  func fromU(_ u: Float) -> Float {
    let s = 1 / (1 + Foundation.exp(-u))
    switch squash {
    case .raw: return u
    case .lin: return lo + (hi - lo) * s
    case .log: return Foundation.exp(Foundation.log(lo) + (Foundation.log(hi) - Foundation.log(lo)) * s)
    }
  }
}

struct RowLayout: Codable, Equatable {
  let name: String
  let segments: [RowSegment]
  var width: Int { segments.last.map { $0.offset + $0.width } ?? 0 }
  func segment(_ name: String) -> RowSegment? { segments.first { $0.name == name } }
}

struct LearnableTensorInfo: Equatable {
  /// `@name`, else the file name without extension.
  let name: String
  let shape: [Int]
  let file: String?
  let initKind: String
}

struct LatentTableInfo: Equatable {
  let name: String
  let dim: Int
}

struct ParamRowInfo: Equatable {
  let name: String
  let layout: String
}

/// Raised by `param-row` in `.capture` mode once the row is evaluated, so a decoder-only pass
/// does not build the voice graph.
struct DatasetCaptureDone: Error {}

/// What the trainer attaches to an evaluator. Leaves are owned by the trainer and survive graph
/// clears; the evaluator only asks for them by name.
final class DatasetContext {
  enum RowMode {
    /// Evaluate the decoder; `param-row` records U and returns it.
    case decode
    /// Evaluate the decoder, record U, and stop evaluation (`DatasetCaptureDone`).
    case capture
    /// `param-row` returns this [B, P] tensor and does not evaluate its argument.
    case leaf(Tensor)
  }

  /// What one-shot latches do (§ one-shot lowering). Latches are the instrument's
  /// static/per-sample boundary, so the trainer cuts the graph there: the voice is rendered from
  /// leaves holding the latched values (gradients arrive already summed over frames), and the
  /// static side is backpropagated separately. Backpropagating a per-frame adjoint through the
  /// static ops instead tapes [frames, B, P] per use.
  enum LatchMode {
    /// Return the latched value itself.
    case identity
    /// Record each static latched tensor (in evaluation order) and return it.
    case capture
    /// Return the k-th leaf for the k-th static latched tensor.
    case leaves([Tensor])
  }

  var latchMode: LatchMode = .identity
  /// Static latched tensors in evaluation order (`.capture`).
  var latched: [Tensor] = []
  var latchIndex = 0

  /// B: rows in this evaluation.
  let rows: Int
  /// Frames rendered (the `hit-time` ramp never wraps inside them).
  let frames: Int
  let sampleRate: Float
  var rowMode: RowMode
  /// name → U, filled by `param-row` in `.decode` / `.capture`.
  var captured: [String: Tensor] = [:]
  /// name, shape, data loaded from the patch's file (nil when absent), @init → leaf.
  let learnable: (String, [Int], [Float]?, String) throws -> Tensor
  /// name, dim → [B, D] code rows.
  let latentRows: (String, Int) throws -> Tensor

  init(
    rows: Int, frames: Int, sampleRate: Float, rowMode: RowMode,
    learnable: @escaping (String, [Int], [Float]?, String) throws -> Tensor,
    latentRows: @escaping (String, Int) throws -> Tensor
  ) {
    self.rows = rows
    self.frames = frames
    self.sampleRate = sampleRate
    self.rowMode = rowMode
    self.learnable = learnable
    self.latentRows = latentRows
  }
}

extension LispEvaluator {
  // MARK: latent-table

  func evalLatentTable(_ args: [ASTNode], attributes: [(name: String, value: String)]) throws
    -> EvalResult
  {
    guard args.count <= 1 else {
      throw LispError.invalidArgument("latent-table takes one default expression: (latent-table @name N @dim D X)")
    }
    let name = attrValue(attributes, "@name").map(unquote) ?? "codes"
    guard let dim = attrValue(attributes, "@dim").flatMap(Int.init), dim > 0 else {
      throw LispError.invalidArgument("latent-table requires @dim N (N > 0)")
    }
    if !latentTables.contains(where: { $0.name == name }) {
      latentTables.append(LatentTableInfo(name: name, dim: dim))
    }
    if let dataset {
      let rows = try dataset.latentRows(name, dim)
      guard rows.shape == [dataset.rows, dim] else {
        throw LispError.invalidArgument(
          "latent-table \(name): trainer supplied \(rows.shape), expected [\(dataset.rows), \(dim)]")
      }
      return .tensor(rows)
    }
    guard let defaultNode = args.first else {
      throw LispError.invalidArgument("latent-table \(name) needs a default [1,\(dim)] expression")
    }
    let value = try evaluateAST(defaultNode)
    guard let shape = tensorShape(of: value), shape == [1, dim] else {
      throw LispError.invalidArgument(
        "latent-table \(name) default must be [1, \(dim)], got \(tensorShape(of: value) ?? [])")
    }
    return value
  }

  // MARK: learnable tensors

  /// `(tensor … @learnable true)`: a constant in compiles, a leaf under a dataset context.
  func evalLearnableTensor(_ args: [ASTNode], attributes: [(name: String, value: String)]) throws
    -> EvalResult
  {
    let shape = try parseShapeFromArgsOrAttributes(args, attributes: attributes, op: "tensor")
    let file = attrValue(attributes, "@file").map(unquote)
    let name =
      attrValue(attributes, "@name").map(unquote)
      ?? file.map { URL(fileURLWithPath: $0).deletingPathExtension().lastPathComponent }
    guard let name else {
      throw LispError.invalidArgument("a @learnable tensor needs @name or @file")
    }
    let initKind = attrValue(attributes, "@init").map(unquote) ?? "glorot"
    guard ["glorot", "zeros", "small", "file"].contains(initKind) else {
      throw LispError.invalidArgument("@init must be glorot, zeros, small or file (got \(initKind))")
    }
    if !learnableTensors.contains(where: { $0.name == name }) {
      learnableTensors.append(
        LearnableTensorInfo(name: name, shape: shape, file: file, initKind: initKind))
    }
    guard let dataset else {
      var plain = attributes
      plain.removeAll { $0.name == "@learnable" || $0.name == "@init" }
      return try evalWavetable(args, attributes: plain, mutable: false)
    }
    var fileData: [Float]? = nil
    if let file {
      let url = URL(fileURLWithPath: file, relativeTo: sourceDirectory).standardizedFileURL
      if FileManager.default.fileExists(atPath: url.path) {
        fileData = try loadTensorData(file: file, expectedShape: shape)
      }
    }
    let leaf = try dataset.learnable(name, shape, fileData, initKind)
    guard leaf.shape == shape else {
      throw LispError.invalidArgument("learnable \(name): trainer supplied \(leaf.shape), expected \(shape)")
    }
    return .tensor(leaf)
  }

  // MARK: row layouts

  func evalRowLayout(_ args: [ASTNode]) throws -> EvalResult {
    guard case .atom(let name)? = args.first else {
      throw LispError.invalidArgument("row-layout requires a name: (row-layout NAME (SEG width squash lo hi) …)")
    }
    var segments: [RowSegment] = []
    var offset = 0
    for node in args.dropFirst() {
      guard case .list(let parts) = node, parts.count >= 3,
        case .atom(let seg) = parts[0], case .atom(let widthText) = parts[1],
        case .atom(let squashText) = parts[2]
      else {
        throw LispError.invalidArgument("row-layout \(name): each segment is (SEG width squash [lo hi])")
      }
      guard let width = Int(widthText), width > 0 else {
        throw LispError.invalidArgument("row-layout \(name) \(seg): width must be a positive integer")
      }
      guard let squash = RowSquash(rawValue: squashText) else {
        throw LispError.invalidArgument("row-layout \(name) \(seg): squash must be log, lin or raw")
      }
      var lo: Float = 0
      var hi: Float = 0
      if squash != .raw {
        guard parts.count == 5 else {
          throw LispError.invalidArgument("row-layout \(name) \(seg): \(squash) needs lo and hi")
        }
        lo = try requireFloat(evaluateAST(parts[3]))
        hi = try requireFloat(evaluateAST(parts[4]))
        guard hi > lo, squash != .log || lo > 0 else {
          throw LispError.invalidArgument("row-layout \(name) \(seg): need hi > lo (and lo > 0 for log)")
        }
      }
      guard !segments.contains(where: { $0.name == seg }) else {
        throw LispError.invalidArgument("row-layout \(name): duplicate segment \(seg)")
      }
      segments.append(RowSegment(name: seg, offset: offset, width: width, squash: squash, lo: lo, hi: hi))
      offset += width
    }
    guard !segments.isEmpty else {
      throw LispError.invalidArgument("row-layout \(name) has no segments")
    }
    let layout = RowLayout(name: name, segments: segments)
    if let existing = rowLayouts[name], existing != layout {
      throw LispError.invalidArgument("row-layout \(name) redefined differently")
    }
    rowLayouts[name] = layout
    return .none
  }

  func evalSeg(_ args: [ASTNode]) throws -> EvalResult {
    guard args.count == 3, case .atom(let layoutName) = args[1], case .atom(let segName) = args[2]
    else {
      throw LispError.invalidArgument("seg requires (seg U LAYOUT SEGMENT)")
    }
    guard let layout = rowLayouts[layoutName] else {
      throw LispError.invalidArgument("seg: unknown row-layout \(layoutName)")
    }
    guard let seg = layout.segment(segName) else {
      throw LispError.invalidArgument("seg: row-layout \(layoutName) has no segment \(segName)")
    }
    let value = try evaluateAST(args[0])
    guard let shape = tensorShape(of: value), shape.count == 2, shape[1] == layout.width else {
      throw LispError.invalidArgument(
        "seg: U must be [B, \(layout.width)] for row-layout \(layoutName), got \(tensorShape(of: value) ?? [])")
    }
    let range: [(Int, Int)?] = [(0, shape[0]), (seg.offset, seg.offset + seg.width)]
    switch value {
    case .tensor(let u): return .tensor(squash(u.shrink(range), seg))
    case .signalTensor(let u): return .signalTensor(squash(u.shrink(range), seg))
    default: throw LispError.typeError("seg: U must be a tensor or signalTensor")
    }
  }

  private func squash(_ x: Tensor, _ seg: RowSegment) -> Tensor {
    switch seg.squash {
    case .raw: return x
    case .lin: return x.sigmoid() * (seg.hi - seg.lo) + seg.lo
    case .log:
      let a = Foundation.log(seg.lo)
      return (x.sigmoid() * (Foundation.log(seg.hi) - a) + a).exp()
    }
  }

  private func squash(_ x: SignalTensor, _ seg: RowSegment) -> SignalTensor {
    switch seg.squash {
    case .raw: return x
    case .lin: return sigmoid(x) * (seg.hi - seg.lo) + seg.lo
    case .log:
      let a = Foundation.log(seg.lo)
      return exp(sigmoid(x) * (Foundation.log(seg.hi) - a) + a)
    }
  }

  // MARK: param-row

  func evalParamRow(_ args: [ASTNode], attributes: [(name: String, value: String)]) throws
    -> EvalResult
  {
    guard args.count == 1 else {
      throw LispError.invalidArgument("param-row requires (param-row U @layout NAME)")
    }
    guard let layoutName = attrValue(attributes, "@layout"), let layout = rowLayouts[layoutName]
    else {
      throw LispError.invalidArgument("param-row requires @layout naming a row-layout defined earlier")
    }
    let name = attrValue(attributes, "@name").map(unquote) ?? "row"
    if !paramRows.contains(where: { $0.name == name }) {
      paramRows.append(ParamRowInfo(name: name, layout: layoutName))
    }
    if let dataset, case .leaf(let leaf) = dataset.rowMode {
      guard leaf.shape == [dataset.rows, layout.width] else {
        throw LispError.invalidArgument(
          "param-row \(name): leaf is \(leaf.shape), expected [\(dataset.rows), \(layout.width)]")
      }
      return .tensor(leaf)
    }
    let value = try evaluateAST(args[0])
    guard let shape = tensorShape(of: value), shape.count == 2, shape[1] == layout.width else {
      throw LispError.invalidArgument(
        "param-row \(name): U must be [B, \(layout.width)], got \(tensorShape(of: value) ?? [])")
    }
    if let dataset {
      guard case .tensor(let u) = value else {
        throw LispError.invalidArgument(
          "param-row \(name): under training U must be static (decoded from codes and weights only)")
      }
      dataset.captured[name] = u
      if case .capture = dataset.rowMode { throw DatasetCaptureDone() }
    }
    return value
  }

  // MARK: row-out

  func evalRowOut(_ args: [ASTNode]) throws -> EvalResult {
    guard args.count == 1 else {
      throw LispError.invalidArgument("row-out requires 1 argument")
    }
    let value = try evaluateAST(args[0])
    if let dataset {
      let lanes = dataset.rows
      switch value {
      case .signalTensor(let y):
        guard y.shape.reduce(1, *) == lanes, y.shape.first == lanes else {
          throw LispError.invalidArgument("row-out: expected [\(lanes)] or [\(lanes),1], got \(y.shape)")
        }
        return .signalTensor(y.shape.count == 1 ? y : y.reshape([lanes]))
      case .signal(let s):
        return .signalTensor(Tensor([Float](repeating: 1, count: lanes)) * s)
      case .float(let f):
        return .signalTensor(Tensor([Float](repeating: 1, count: lanes)) * Signal.constant(f))
      default:
        throw LispError.typeError("row-out: value must be a signal or a [B] signal tensor")
      }
    }
    switch value {
    case .signal, .float: return value
    case .signalTensor(let y):
      guard y.shape.reduce(1, *) == 1 else {
        throw LispError.invalidArgument("row-out: an instrument voice has one row; got \(y.shape)")
      }
      return .signal(y.sum())
    case .tensor(let t):
      guard t.shape.reduce(1, *) == 1 else {
        throw LispError.invalidArgument("row-out: an instrument voice has one row; got \(t.shape)")
      }
      return .signal(Signal.constant(0) + t.peek(Signal.constant(0)))
    default:
      throw LispError.typeError("row-out: value must be a signal or a one-row tensor")
    }
  }

  // MARK: hit-time

  func evalHitTime(_ args: [ASTNode], attributes: [(name: String, value: String)]) throws
    -> EvalResult
  {
    guard args.count == 1 else {
      throw LispError.invalidArgument("hit-time requires (hit-time TRIGGER [@max seconds])")
    }
    let maxSeconds = attrValue(attributes, "@max").flatMap(Float.init) ?? 8
    if let dataset {
      // Frame-parallel seconds since frame 0: a deterministic phasor at sr/(2N) never wraps
      // inside N frames.
      let n = Float(dataset.frames)
      let ramp = Signal.phasor(Tensor([dataset.sampleRate / (2 * n)])).sum()
      return .signal(ramp * (2 * n / dataset.sampleRate))
    }
    // seconds since the last rising edge (0 on the edge frame); maxSeconds before the first
    // edge and once that much time has passed.
    macroExpansionCounter += 1
    let p = "__hit_time_\(macroExpansionCounter)_"
    let source = """
      (make-history \(p)prev)
      (make-history \(p)elapsed)
      (make-history \(p)armed)
      (def \(p)trig \(p)arg)
      (def \(p)high (gt \(p)trig 0.5))
      (def \(p)edge (* \(p)high (lte (read-history \(p)prev) 0.5)))
      (write-history \(p)prev \(p)high)
      (def \(p)played (max \(p)edge (read-history \(p)armed)))
      (write-history \(p)armed \(p)played)
      (def \(p)n (gswitch \(p)edge 0 (min (* \(maxSeconds) samplerate) (+ 1 (read-history \(p)elapsed)))))
      (write-history \(p)elapsed \(p)n)
      (def \(p)out (gswitch \(p)played (/ \(p)n samplerate) \(maxSeconds)))
      """
    definitions[p + "arg"] = try evaluateAST(args[0])
    for node in try parseSource(source) { _ = try evaluateAST(node) }
    guard let out = definitions[p + "out"] else {
      throw LispError.invalidArgument("hit-time: internal expansion failed")
    }
    return out
  }

  // MARK: declick

  /// `(declick Y TRIGGER)`: a retrigger cuts the previous hit; carry its last output sample and
  /// let it fall over ~1 ms so the cut does not click. Identity under a dataset context (one hit
  /// per row; the carry would feed the output back through a history).
  func evalDeclick(_ args: [ASTNode]) throws -> EvalResult {
    guard args.count == 2 else {
      throw LispError.invalidArgument("declick requires (declick SIGNAL TRIGGER)")
    }
    if dataset != nil { return try evaluateAST(args[0]) }
    macroExpansionCounter += 1
    let p = "__declick_\(macroExpansionCounter)_"
    definitions[p + "y"] = try evaluateAST(args[0])
    definitions[p + "trig"] = try evaluateAST(args[1])
    let source = """
      (make-history \(p)prev)
      (make-history \(p)last)
      (make-history \(p)carry)
      (def \(p)high (gt \(p)trig 0.5))
      (def \(p)edge (* \(p)high (lte (read-history \(p)prev) 0.5)))
      (write-history \(p)prev \(p)high)
      (def \(p)c (gswitch \(p)edge (read-history \(p)last) (* (read-history \(p)carry) (exp (/ -1000 samplerate)))))
      (write-history \(p)carry \(p)c)
      (def \(p)out (+ \(p)y \(p)c))
      (write-history \(p)last \(p)out)
      """
    for node in try parseSource(source) { _ = try evaluateAST(node) }
    return definitions[p + "out"]!
  }

  // MARK: one-shot lowering of hit-clock latches

  /// Under a dataset context, `latch`/`event-hold`/`event-latch` of a static value is the value.
  func oneShotLatch(_ op: String, _ args: [ASTNode]) throws -> EvalResult? {
    guard dataset != nil else { return nil }
    guard args.count == 2 else {
      throw LispError.invalidArgument("\(op) requires 2 arguments (value, trigger)")
    }
    let value = try evaluateAST(args[0])
    switch value {
    case .tensor(let t):
      guard let dataset else { return value }
      let k = dataset.latchIndex
      dataset.latchIndex += 1
      switch dataset.latchMode {
      case .identity: return value
      case .capture:
        dataset.latched.append(t)
        return value
      case .leaves(let leaves):
        guard k < leaves.count, leaves[k].shape == t.shape else {
          throw LispError.invalidArgument(
            "\(op): latch \(k) has shape \(t.shape) but the trainer's leaf does not match")
        }
        return .tensor(leaves[k])
      }
    case .float: return value
    case .signal(let s) where s.data != nil: return value
    default:
      throw LispError.invalidArgument(
        "\(op): under dataset training the latched value must be static (one hit at frame 0); "
          + "this one varies per frame")
    }
  }
}
