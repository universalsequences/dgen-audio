import Foundation

public enum DGenBinaryAudit {
  /// Repository-relative audit script for this platform's object format.
  /// Mach-O (`.dylib`) on Apple platforms, ELF (`.so`) everywhere else. Only
  /// the last, repo-derived tier of the resolution order is platform-varying;
  /// an explicit `--audit-tool` path or `DGEN_BINARY_AUDIT_TOOL` is honoured
  /// verbatim on every platform.
  static let repositoryScriptName: String = {
    #if canImport(Darwin)
      return "audit-dgen-dylib.sh"
    #else
      return "audit-dgen-elf-so.sh"
    #endif
  }()

  /// Resolve the audit script. Precedence: explicit tool path (CLI
  /// `--audit-tool`) > `DGEN_BINARY_AUDIT_TOOL` env var > a `scripts/`
  /// directory beside the executable. No fallback may derive from `#filePath`:
  /// that would embed the checkout used to build a published binary.
  static func executableRelativeScript(executableDirectory: URL) -> String {
    executableDirectory.appendingPathComponent("scripts/\(repositoryScriptName)").path
  }

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
      script = executableRelativeScript(
        executableDirectory: DGenToolchainPolicy.executableDirectory)
      resolvedFromRepoFallback = true
    }

    guard FileManager.default.isExecutableFile(atPath: script) else {
      let message: String
      if resolvedFromRepoFallback {
        message = """
          DGen binary audit script not found at the executable-relative path: \
          \(script). Install scripts/ beside DGenLisp, pass --audit-tool \
          <path-to-\(repositoryScriptName)>, set DGEN_BINARY_AUDIT_TOOL, or \
          pass --skip-inline-audit if the host application audits compiled \
          dylibs itself.
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
        userInfo: [NSLocalizedDescriptionKey: "DGen binary audit failed: \(output)"])
    }
  }
}
