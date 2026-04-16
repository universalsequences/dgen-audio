import Foundation

struct ModelCheckpoint: Codable {
  var step: Int
  var createdAtUTC: String
  var params: [NamedTensorSnapshot]
}

enum CheckpointStore {
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
    try data.write(to: checkpointsDir.appendingPathComponent(fileName))
  }

  static func writeBestModelState(
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
    try data.write(to: checkpointsDir.appendingPathComponent("model_best.json"))
  }
}
