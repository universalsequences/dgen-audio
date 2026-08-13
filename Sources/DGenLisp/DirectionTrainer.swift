// DirectionTrainer.swift — E4 direction-finding mode (SUBTRACTIVE_SPEC E4,
// patch-learn-spec §3): seeded short run + one cold restart as basin check.
//
// Phase C wires this; until then any non-plan-only real job fails with a
// protocol-clean error event.

import DGen
import DGenLazy
import DGenTrainProtocol
import Foundation

enum DirectionTrainer {
    static func train(
        options: TrainOptions,
        patchSource: String,
        patchPlan: PatchPlan,
        targetSamples: [Float],
        targetSampleRate: Float,
        sink: TrainEventSink,
        jobDir: JobDir
    ) throws -> ResultEvent {
        throw TrainProtocolError(
            "direction-mode training not yet implemented; use --plan-only or --fake-trainer")
    }
}
