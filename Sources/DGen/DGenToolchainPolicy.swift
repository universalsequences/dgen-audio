import CryptoKit
import Foundation

public struct DGenCompilerInvocation {
  public let executable: String
  public let arguments: [String]
  public let policySignature: String
}

/// Versioned compile/link policy shared by the DGen runtime and DGenLisp.
///
/// Production selects the staged toolchain explicitly: hosts pass the stage
/// root per invocation (`--toolchain-root`), and `DGEN_TOOLCHAIN_STAGE_ROOT`
/// remains as a development fallback. The system-Clang path is a
/// development-only compatibility path; generated C and numerical semantics
/// are identical.
public enum DGenToolchainPolicy {
  public static let policyVersion = 1
  public static let target = "arm64-apple-macos11.0"

  private static let optimizationArguments = [
    // `-Ofast` is deprecated. DGen spells out its numerical contract:
    // aggressive O3 optimization plus finite-only/unsafe algebra. Boundary
    // containment remains valid because dgen_runtime.h classifies IEEE-754
    // exponent bits instead of using a classification builtin that these
    // flags may fold away. The Phase 2 NaN/Inf fixture proves this policy.
    "-O3",
    "-ffast-math",
    "-fno-math-errno",
    "-fno-trapping-math",
    "-ffp-contract=fast",
    "-fvectorize",
    "-fslp-vectorize",
    "-funroll-loops",
  ]

  private static let contractArguments = [
    "-mcpu=apple-m1",
    "-flto=thin",
    "-fPIC",
    "-fvisibility=hidden",
    "-fno-stack-protector",
    "-fno-asynchronous-unwind-tables",
    "-std=c11",
    "-x", "c",
  ]

  public static var repositoryRoot: URL {
    URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent() // DGen
      .deletingLastPathComponent() // Sources
      .deletingLastPathComponent() // repository root
  }

  public static var developmentRuntimeInclude: URL {
    if let override = ProcessInfo.processInfo.environment["DGEN_RUNTIME_INCLUDE"],
      !override.isEmpty
    {
      return URL(fileURLWithPath: override)
    }
    return repositoryRoot.appendingPathComponent("toolchain/include", isDirectory: true)
  }

  /// Resolves the staged toolchain root for one invocation.
  ///
  /// An explicit root — the host-selected `--toolchain-root` — always wins over
  /// the `DGEN_TOOLCHAIN_STAGE_ROOT` development fallback. `nil` means no
  /// staged toolchain was selected at all, which is the only case that may use
  /// the system-Clang development path.
  public static func resolvedStageRoot(explicit: String? = nil) -> URL? {
    if let explicit, !explicit.isEmpty {
      return URL(fileURLWithPath: explicit, isDirectory: true)
    }
    if let stagePath = ProcessInfo.processInfo.environment["DGEN_TOOLCHAIN_STAGE_ROOT"],
      !stagePath.isEmpty
    {
      return URL(fileURLWithPath: stagePath, isDirectory: true)
    }
    return nil
  }

  public static func compileInvocation(
    outputPath: String,
    sourcePath: String,
    toolchainRoot: String? = nil
  ) throws -> DGenCompilerInvocation {
    // A selected root is binding: an incomplete stage is an error, never a
    // silent downgrade to the system compiler.
    if let stageRoot = resolvedStageRoot(explicit: toolchainRoot) {
      return try embeddedInvocation(
        stageRoot: stageRoot,
        outputPath: outputPath,
        sourcePath: sourcePath)
    }
    return systemDevelopmentInvocation(outputPath: outputPath, sourcePath: sourcePath)
  }

  /// SHA-256 of the staged distribution's `VERSION.json` — the toolchain's own
  /// identity record, and the only compiler fingerprint the embedded path may
  /// consult. It covers the distribution, ABI, codegen-policy, and LLVM
  /// versions plus the staged `clang`/`lld`/runtime-header digests.
  public static func stagedVersionDigest(stageRoot: URL) -> String {
    let versionFile = stageRoot.appendingPathComponent("VERSION.json")
    guard let data = try? Data(contentsOf: versionFile) else {
      return "unavailable"
    }
    return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
  }

  public static func systemDevelopmentInvocation(
    outputPath: String,
    sourcePath: String
  ) -> DGenCompilerInvocation {
    let arguments =
      ["-target", target]
      + optimizationArguments
      + contractArguments
      + [
        "-I", developmentRuntimeInclude.path,
        "-dynamiclib",
        "-Wl,-dead_strip",
        "-Wl,-install_name,@rpath/\(URL(fileURLWithPath: outputPath).lastPathComponent)",
        "-o", outputPath,
        sourcePath,
      ]
    return DGenCompilerInvocation(
      executable: "/usr/bin/clang",
      arguments: arguments,
      policySignature: signature(mode: "system-development", executable: "/usr/bin/clang",
        arguments: arguments))
  }

  public static func embeddedInvocation(
    stageRoot: URL,
    outputPath: String,
    sourcePath: String
  ) throws -> DGenCompilerInvocation {
    let clang = stageRoot.appendingPathComponent("bin/dgen-clang").path
    let linker = stageRoot.appendingPathComponent("bin/ld64.lld").path
    let resourceDirectory = stageRoot.appendingPathComponent("lib/clang/20").path
    let builtins = stageRoot
      .appendingPathComponent("lib/clang/20/lib/darwin/libclang_rt.builtins.a").path
    let include = stageRoot.appendingPathComponent("include").path
    let stubDirectory = stageRoot.appendingPathComponent("lib").path
    let emptySDK = stageRoot.appendingPathComponent("empty-sdk").path

    // Every file the staged layout promises (toolchain/LAYOUT.md). A selected
    // root that cannot satisfy it fails the compile outright.
    let missing = [
      clang, linker, builtins,
      stageRoot.appendingPathComponent("include/dgen_runtime.h").path,
      stageRoot.appendingPathComponent("lib/libSystem.tbd").path,
      stageRoot.appendingPathComponent("VERSION.json").path,
    ].filter { !FileManager.default.fileExists(atPath: $0) }

    if !missing.isEmpty {
      throw NSError(
        domain: "DGenToolchainPolicy",
        code: 1,
        userInfo: [
          NSLocalizedDescriptionKey: """
            Embedded DGen toolchain root is incomplete: \(stageRoot.path)
            Missing: \(missing.joined(separator: ", "))
            """
        ])
    }

    try FileManager.default.createDirectory(
      atPath: emptySDK,
      withIntermediateDirectories: true)

    let arguments =
      ["-target", target]
      + optimizationArguments
      + contractArguments
      + [
        "-ffreestanding",
        "-nostdinc",
        "-isysroot", emptySDK,
        "-resource-dir", resourceDirectory,
        "-isystem", "\(resourceDirectory)/include",
        "-I", include,
        "-fuse-ld=\(linker)",
        "-nostdlib",
        "-dynamiclib",
        "-Wl,-install_name,@rpath/\(URL(fileURLWithPath: outputPath).lastPathComponent)",
        "-Wl,-undefined,error",
        "-Wl,-fatal_warnings",
        "-Wl,-dead_strip",
        "-L\(stubDirectory)",
        sourcePath,
        "-x", "none",
        builtins,
        "-lSystem",
        "-o", outputPath,
      ]
    return DGenCompilerInvocation(
      executable: clang,
      arguments: arguments,
      policySignature: signature(mode: "embedded", executable: clang, arguments: arguments))
  }

  private static func signature(
    mode: String,
    executable: String,
    arguments: [String]
  ) -> String {
    ([
      "dgen-toolchain-policy=\(policyVersion)",
      "mode=\(mode)",
      "executable=\(executable)",
    ] + arguments).joined(separator: "\n")
  }
}
