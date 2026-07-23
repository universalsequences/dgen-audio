import Foundation

public struct DGenCompilerInvocation {
  public let executable: String
  public let arguments: [String]
  public let policySignature: String
}

/// Versioned compile/link policy shared by the DGen runtime and DGenLisp.
///
/// Production selects the staged toolchain explicitly with
/// `DGEN_TOOLCHAIN_STAGE_ROOT`. The system-Clang path is a development-only
/// compatibility path; generated C and numerical semantics are identical.
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

  public static func compileInvocation(
    outputPath: String,
    sourcePath: String
  ) throws -> DGenCompilerInvocation {
    if let stagePath = ProcessInfo.processInfo.environment["DGEN_TOOLCHAIN_STAGE_ROOT"],
      !stagePath.isEmpty
    {
      return try embeddedInvocation(
        stageRoot: URL(fileURLWithPath: stagePath),
        outputPath: outputPath,
        sourcePath: sourcePath)
    }
    return systemDevelopmentInvocation(outputPath: outputPath, sourcePath: sourcePath)
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

    for required in [
      clang, linker, builtins,
      stageRoot.appendingPathComponent("include/dgen_runtime.h").path,
      stageRoot.appendingPathComponent("lib/libSystem.tbd").path,
    ] where !FileManager.default.fileExists(atPath: required) {
      throw NSError(
        domain: "DGenToolchainPolicy",
        code: 1,
        userInfo: [NSLocalizedDescriptionKey: "Embedded DGen toolchain file is missing: \(required)"])
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
