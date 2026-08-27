#if canImport(CryptoKit)
import CryptoKit
#else
import Crypto
#endif
import Foundation

/// Stable key for compiled DGen audio artifacts.
///
/// This cache intentionally sits *after* gen direct execution has produced a fresh
/// `Graph`, so callers can still collect/register per-instance runtime state
/// (params, tensor/cell IDs, etc.) while avoiding the expensive graph -> C source
/// compilation pipeline when generated code would be identical.
public struct DGenArtifactKey: Hashable {
  public let cacheVersion: Int
  public let graphFingerprint: String
  public let backend: String
  public let frameCount: Int
  public let voiceCount: Int
  public let voiceCellId: Int?
  public let forceScalar: Bool
  public let enableBufferReuse: Bool
  public let gemmStrategy: String
  public let operatorName: String

  public init(
    cacheVersion: Int = DGenArtifactCache.cacheVersion,
    graphFingerprint: String,
    backend: Backend,
    frameCount: Int,
    voiceCount: Int,
    voiceCellId: Int?,
    forceScalar: Bool,
    enableBufferReuse: Bool,
    gemmStrategy: GEMMStrategy,
    operatorName: String
  ) {
    self.cacheVersion = cacheVersion
    self.graphFingerprint = graphFingerprint
    self.backend = String(describing: backend)
    self.frameCount = frameCount
    self.voiceCount = voiceCount
    self.voiceCellId = voiceCellId
    self.forceScalar = forceScalar
    self.enableBufferReuse = enableBufferReuse
    self.gemmStrategy = String(describing: gemmStrategy)
    self.operatorName = operatorName
  }
}

/// Reusable product of `CompilationPipeline.compile` plus dylib-cache metadata.
///
/// A cache hit should still instantiate a fresh `CCompiledKernel` and register it
/// against the current parent node; only the expensive pipeline/source-generation
/// work is reused.
public struct DGenCompiledArtifact {
  public let compilationResult: CompilationResult
  public let source: String
  public let memorySize: Int

  public init(
    compilationResult: CompilationResult,
    source: String,
    memorySize: Int
  ) {
    self.compilationResult = compilationResult
    self.source = source
    self.memorySize = memorySize
  }
}

public final class DGenArtifactCache {
  public static let shared = DGenArtifactCache()

  /// Bump when DGen/codegen semantics change in a way the graph fingerprint does
  /// not capture.
  public static let cacheVersion = 1

  /// Enabled by default. Set `PE_DGEN_ARTIFACT_CACHE=0` or `false` to disable
  /// without changing code while comparing preset timing traces.
  public static var isEnabled: Bool = {
    let value = ProcessInfo.processInfo.environment["PE_DGEN_ARTIFACT_CACHE"] ?? ""
    return !(value == "0" || value.lowercased() == "false")
  }()

  private let lock = NSLock()
  private var storage: [DGenArtifactKey: DGenCompiledArtifact] = [:]
  private(set) public var hitCount: Int = 0
  private(set) public var missCount: Int = 0

  private init() {}

  public func artifact(for key: DGenArtifactKey) -> DGenCompiledArtifact? {
    lock.lock()
    defer { lock.unlock() }
    if let artifact = storage[key] {
      hitCount += 1
      return artifact
    }
    missCount += 1
    return nil
  }

  public func store(_ artifact: DGenCompiledArtifact, for key: DGenArtifactKey) {
    lock.lock()
    storage[key] = artifact
    lock.unlock()
  }

  public func removeAll() {
    lock.lock()
    storage.removeAll(keepingCapacity: true)
    hitCount = 0
    missCount = 0
    lock.unlock()
  }
}

extension Graph {
  /// Fingerprint the generated graph structure and code-affecting metadata.
  /// Runtime values such as current param values are intentionally excluded.
  public func compiledArtifactFingerprint() -> String {
    var lines: [String] = []
    lines.reserveCapacity(nodes.count + tensors.count + 32)

    lines.append("artifactCacheVersion=\(DGenArtifactCache.cacheVersion)")
    lines.append("sampleRate=\(sampleRate)")
    lines.append("maxFrameCount=\(maxFrameCount)")
    lines.append("next=\(next)")
    lines.append("totalMemoryCells=\(totalMemoryCells)")
    lines.append("lazyCells=\(lazyCells.sorted())")
    lines.append("materializeNodes=\(materializeNodes.sorted())")
    lines.append("persistentCells=\(persistentCells.sorted())")
    lines.append("parameterCells=\(parameterCells.sorted())")
    lines.append("gradientSideEffects=\(gradientSideEffects)")
    lines.append("lastForwardNodeId=\(String(describing: lastForwardNodeId))")

    for nodeId in nodes.keys.sorted() {
      guard let node = nodes[nodeId] else { continue }
      lines.append(
        [
          "node", String(node.id),
          "op", String(describing: node.op),
          "inputs", String(describing: node.inputs),
          "temporal", String(describing: node.temporalDependencies),
          "shape", String(describing: node.shape),
        ].joined(separator: "|")
      )
    }

    for tensorId in tensors.keys.sorted() {
      guard let tensor = tensors[tensorId] else { continue }
      lines.append(
        [
          "tensor", String(tensor.id),
          "shape", String(describing: tensor.shape),
          "cell", String(tensor.cellId),
          "baseShape", String(describing: tensor.baseShape),
          "baseStrides", String(describing: tensor.baseStrides),
          "transforms", String(describing: tensor.transforms),
          "isLazy", String(tensor.isLazy),
          "materialize", String(tensor.materialize),
          "hasData", String(tensor.data != nil),
          "dataCount", String(tensor.data?.count ?? 0),
        ].joined(separator: "|")
      )
    }

    lines.append("nodeToTensor=\(nodeToTensor.sorted { $0.key < $1.key }.map { "\($0.key):\($0.value)" }.joined(separator: ","))")
    lines.append("cellToTensor=\(cellToTensor.sorted { $0.key < $1.key }.map { "\($0.key):\($0.value)" }.joined(separator: ","))")
    lines.append("cellAllocationSizes=\(cellAllocationSizes.sorted { $0.key < $1.key }.map { "\($0.key):\($0.value)" }.joined(separator: ","))")
    lines.append("nodeHopRate=\(nodeHopRate.sorted { $0.key < $1.key }.map { "\($0.key):\($0.value.0):\($0.value.1)" }.joined(separator: ","))")
    lines.append("nodePositionDep=\(nodePositionDep.sorted { $0.key < $1.key }.map { "\($0.key):\($0.value)" }.joined(separator: ","))")
    lines.append("gradCarryCells=\(gradCarryCells.sorted { $0.key < $1.key }.map { "\($0.key):\($0.value)" }.joined(separator: ","))")
    lines.append("tensorGradCells=\(tensorGradCells.sorted { $0.key < $1.key }.map { "\($0.key):\($0.value)" }.joined(separator: ","))")
    lines.append("tensorGradCarryCells=\(tensorGradCarryCells.sorted())")
    lines.append("frameAwareCells=\(frameAwareCells.sorted { $0.key < $1.key }.map { "\($0.key):\($0.value.tensorSize):\($0.value.frameCount)" }.joined(separator: ","))")
    lines.append("frameAwareCellHops=\(frameAwareCellHops.sorted { $0.key < $1.key }.map { "\($0.key):\($0.value)" }.joined(separator: ","))")
    lines.append("frameAwareCellScatter=\(frameAwareCellScatter.sorted())")
    lines.append("simdOptimizedConv2Ds=\(simdOptimizedConv2Ds.sorted())")
    lines.append("conv2dMaskCells=\(conv2dMaskCells.sorted { $0.key < $1.key }.map { "\($0.key):\($0.value)" }.joined(separator: ","))")

    let data = lines.joined(separator: "\n").data(using: .utf8) ?? Data()
    let digest = SHA256.hash(data: data)
    return digest.map { String(format: "%02x", $0) }.joined()
  }
}
