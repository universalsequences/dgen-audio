import Foundation

public enum DGenBinaryAudit {
  public static func audit(_ dylibPath: String) throws {
    let script: String
    if let override = ProcessInfo.processInfo.environment["DGEN_BINARY_AUDIT_TOOL"],
      !override.isEmpty
    {
      script = override
    } else {
      script = DGenToolchainPolicy.repositoryRoot
        .appendingPathComponent("scripts/audit-dgen-dylib.sh").path
    }

    guard FileManager.default.isExecutableFile(atPath: script) else {
      throw NSError(
        domain: "DGenBinaryAudit",
        code: 1,
        userInfo: [NSLocalizedDescriptionKey: "DGen binary audit tool is unavailable: \(script)"])
    }

    let process = Process()
    process.executableURL = URL(fileURLWithPath: script)
    process.arguments = [dylibPath]
    let diagnostics = Pipe()
    process.standardOutput = diagnostics
    process.standardError = diagnostics
    try process.run()
    process.waitUntilExit()
    let data = diagnostics.fileHandleForReading.readDataToEndOfFile()
    let output = String(data: data, encoding: .utf8) ?? ""
    guard process.terminationStatus == 0 else {
      throw NSError(
        domain: "DGenBinaryAudit",
        code: Int(process.terminationStatus),
        userInfo: [NSLocalizedDescriptionKey: "DGen dylib audit failed: \(output)"])
    }
  }
}
