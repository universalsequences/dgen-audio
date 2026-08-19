import Foundation

struct ModalDrumCheckpoint: Codable {
  var step: Int
  var loss: Float
  var createdAtUTC: String
  var patch: ModalPatch
}

enum ModalCheckpointStore {
  static func write(_ checkpoint: ModalDrumCheckpoint, to url: URL) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(checkpoint).write(to: url, options: .atomic)
  }
}
