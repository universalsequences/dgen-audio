import Foundation

struct TrainingCheckpoint: Codable {
  var step: Int
  var createdAtUTC: String
  var note: String
  var seenChunks: Int
}

struct ModelCheckpoint: Codable {
  var step: Int
  var createdAtUTC: String
  var params: [NamedTensorSnapshot]
}

enum CheckpointStore {
  static func writePlaceholder(
    checkpointsDir: URL,
    step: Int,
    seenChunks: Int,
    note: String
  ) throws {
    let checkpoint = TrainingCheckpoint(
      step: step,
      createdAtUTC: ISO8601DateFormatter().string(from: Date()),
      note: note,
      seenChunks: seenChunks
    )

    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try encoder.encode(checkpoint)

    let fileName = String(format: "step_%08d.json", step)
    let url = checkpointsDir.appendingPathComponent(fileName)
    try data.write(to: url)
  }

  static func writeModelState(
    checkpointsDir: URL,
    step: Int,
    params: [NamedTensorSnapshot]
  ) throws {
    let checkpoint = ModelCheckpoint(
      step: step,
      createdAtUTC: ISO8601DateFormatter().string(from: Date()),
      params: params
    )

    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try encoder.encode(checkpoint)

    let fileName = String(format: "model_step_%08d.json", step)
    let url = checkpointsDir.appendingPathComponent(fileName)
    try data.write(to: url)
  }
}
