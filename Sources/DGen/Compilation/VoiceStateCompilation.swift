import Foundation

/// Compilation-time plan for MC (multi-channel) voice-index handling.
///
/// A caller (like a patch editor) may launch multiple copies of the same compiled
/// graph (one per voice).
///
/// Each copy writes its voice index into a memory cell so generated C code can:
/// 1) clamp the voice index to `0..<voiceCount`, and
/// 2) derive per-voice scratch/global offsets.
struct VoiceStatePlan {
  /// Total number of MC voices expected at runtime.
  let voiceCount: Int
  /// Logical graph cell containing the runtime voice index.
  let logicalVoiceCellId: CellID?
  /// True when compilation synthesized a dedicated voice-index cell.
  let generatedVoiceCell: Bool

  /// Only synthesized cells need explicit reservation during remapping, because they are not
  /// referenced by any UOp and would otherwise be dropped from `cellMappings`.
  var voiceCellIdForMemoryRemap: CellID? {
    generatedVoiceCell ? logicalVoiceCellId : nil
  }

  /// Resolves the final physical memory cell ID used by generated code.
  ///
  /// Falls back to the logical cell for explicit user-provided cells that were intentionally not
  /// part of remapping inputs.
  func physicalVoiceCellId(_ cellAllocations: CellAllocations) -> CellID? {
    guard let logicalVoiceCellId else { return nil }
    return cellAllocations.cellMappings[logicalVoiceCellId] ?? logicalVoiceCellId
  }
}

/// Namespace for voice-state plumbing used by `CompilationPipeline`.
enum VoiceStateCompilation {
  /// Builds a voice-state plan from compile options.
  ///
  /// - If `voiceCount <= 1`, no voice-index cell is required.
  /// - If `voiceCount > 1` and `options.voiceCellId == nil`, a new graph cell is allocated.
  static func buildPlan(graph: Graph, options: CompilationPipeline.Options) -> VoiceStatePlan {
    var logicalVoiceCellId = options.voiceCellId
    var generatedVoiceCell = false
    if options.voiceCount > 1 && logicalVoiceCellId == nil {
      logicalVoiceCellId = graph.alloc()
      generatedVoiceCell = true
    }
    return VoiceStatePlan(
      voiceCount: options.voiceCount,
      logicalVoiceCellId: logicalVoiceCellId,
      generatedVoiceCell: generatedVoiceCell)
  }

  /// Applies MC voice configuration to renderers that support voice-indexed scratch layout.
  static func configureRenderer(
    _ renderer: Renderer, plan: VoiceStatePlan, cellAllocations: CellAllocations
  ) {
    guard let cRenderer = renderer as? CRenderer else { return }
    cRenderer.voiceCount = plan.voiceCount
    cRenderer.voiceCellIdOpt = plan.physicalVoiceCellId(cellAllocations)
  }
}
