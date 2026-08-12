import Foundation

public enum DGenBinaryAudit {
  /// Resolve the audit script. Precedence: explicit tool path (CLI
  /// `--audit-tool`) > `DGEN_BINARY_AUDIT_TOOL` env var > the repository
  /// script derived from `#filePath`. The last is a build-machine path: it
  /// only exists where dgen itself was checked out, so a shipped binary must
  /// either be told where the tool is or be run with the inline audit
  /// skipped (`--skip-inline-audit`) when the host audits the artifact
  /// itself.
  public static func audit(_ dylibPath: String, auditTool: String? = nil) throws {
    let script: String
    var resolvedFromRepoFallback = false
    if let explicit = auditTool, !explicit.isEmpty {
      script = explicit
    } else if let override = ProcessInfo.processInfo.environment["DGEN_BINARY_AUDIT_TOOL"],
      !override.isEmpty
    {
      script = override
    } else {
      script = DGenToolchainPolicy.repositoryRoot
        .appendingPathComponent("scripts/audit-dgen-dylib.sh").path
      resolvedFromRepoFallback = true
    }

    guard FileManager.default.isExecutableFile(atPath: script) else {
      let message: String
      if resolvedFromRepoFallback {
        message = """
          DGen binary audit script not found: \(script). This default is the \
          build machine's dgen checkout and does not exist here. Pass \
          --audit-tool <path-to-audit-dgen-dylib.sh>, set \
          DGEN_BINARY_AUDIT_TOOL, or pass --skip-inline-audit if the host \
          application audits compiled dylibs itself.
          """
      } else {
        message = "DGen binary audit tool is unavailable or not executable: \(script)"
      }
      throw NSError(
        domain: "DGenBinaryAudit",
        code: 1,
        userInfo: [NSLocalizedDescriptionKey: message])
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
