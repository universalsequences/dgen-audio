#if canImport(CryptoKit)
import CryptoKit
#else
import Crypto
#endif
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
///
/// Everything that varies with the host object format / ISA lives in
/// `HostProfile`, and exactly one profile is selected at compile time. The
/// `darwin-arm64` profile is the original policy verbatim: its emitted
/// argument lists and policy signatures are byte-identical to the pre-port
/// ones, because they are also a compilation-cache key.
public enum DGenToolchainPolicy {
  public static let policyVersion = 1

  // MARK: - Host profiles

  /// The host-varying half of the compile/link policy.
  ///
  /// Only object-format and ISA concerns belong here. The numerical contract
  /// (`optimizationArguments`) and the language/ABI contract shared by every
  /// host (`sharedContractArguments`) are deliberately profile-independent:
  /// generated C and numerical semantics must be identical on every platform.
  struct HostProfile {
    /// Stable profile identity, recorded in the policy signature.
    let identifier: String
    /// Clang target triple.
    let target: String
    /// ISA baseline. See the `linux-x86_64` profile for why v3 is a floor.
    let cpuArguments: [String]
    /// Extra compile-side flags needed for this host's dead-strip mechanism.
    let sectionArguments: [String]
    /// Driver flag that produces a shared library.
    let sharedLibraryArgument: String
    /// Linker dead-code stripping.
    let deadStripArgument: String
    /// File extension of the produced artifact, without the leading dot.
    let artifactExtension: String
    /// Whether a staged (hermetic) toolchain exists for this host at all.
    let supportsStagedToolchain: Bool
    /// Extra leading lines mixed into the policy signature.
    ///
    /// Empty for `darwin-arm64`: that signature predates host profiles and is
    /// kept byte-identical so existing artifact caches stay valid. Non-darwin
    /// profiles carry an explicit identity line instead of relying on the
    /// argument list alone to distinguish them.
    let signaturePreamble: [String]
    /// Linker flag that stamps the artifact's own name into it — Mach-O
    /// `LC_ID_DYLIB` / ELF `DT_SONAME`. Must be a bare filename plus, on
    /// Darwin, the `@rpath` prefix the mac host expects.
    let installNameArgument: (String) -> String

    // MARK: Staged-toolchain layout
    //
    // The staged layout is object-format specific (toolchain/LAYOUT.md). These
    // fields keep `embeddedInvocation` profile-driven rather than branching on
    // the host inside it, which is what let the Mach-O argument list stay
    // byte-identical while the ELF one was added.

    /// Stage-relative path to the linker selected with `-fuse-ld=`.
    let stageLinkerRelativePath: String
    /// Stage-relative path to the compiler-rt builtins archive, linked
    /// explicitly because the link is `-nostdlib`.
    let stageBuiltinsRelativePath: String
    /// Stage-relative paths preflighted in addition to the linker, the builtins
    /// archive, `include/dgen_runtime.h`, and `VERSION.json`.
    let stageAdditionalRequiredRelativePaths: [String]
    /// Stage-relative directory passed as `-isysroot`, or `nil` when the object
    /// format has no sysroot to neutralise. Mach-O passes a deliberately empty
    /// one so no system SDK can leak in; ELF has no equivalent knob, and
    /// `-nostdinc` plus the pinned `-resource-dir` already close that door.
    let stageEmptySDKRelativePath: String?
    /// Stage-relative directory searched for the stub system library, or `nil`
    /// when the profile links against no stub.
    let stageStubLibraryRelativeDirectory: String?
    /// Link arguments for the stage's stub system library.
    let stubLibraryLinkArguments: [String]
    /// Linker strictness for symbols left undefined by the link.
    ///
    /// Mach-O resolves the whole permitted surface through `libSystem.tbd` at
    /// link time, so anything still undefined is a real error. An ELF
    /// `-shared -nostdlib` object is the opposite case: its libc/libm surface
    /// is *meant* to stay undefined for the dynamic loader to bind at
    /// `dlopen`, and that surface is exactly what `abi/libsystem-symbols-v1-elf.txt`
    /// allowlists and `scripts/audit-dgen-elf-so.sh` verifies after the fact.
    /// Demanding a fully-resolved link there would reject every valid artifact.
    let undefinedSymbolArguments: [String]
  }

  #if os(macOS) && arch(arm64)

    static let hostProfile = HostProfile(
      identifier: "darwin-arm64",
      target: "arm64-apple-macos11.0",
      cpuArguments: ["-mcpu=apple-m1"],
      sectionArguments: [],
      sharedLibraryArgument: "-dynamiclib",
      deadStripArgument: "-Wl,-dead_strip",
      artifactExtension: "dylib",
      supportsStagedToolchain: true,
      signaturePreamble: [],
      installNameArgument: { base in "-Wl,-install_name,@rpath/\(base)" },
      stageLinkerRelativePath: "bin/ld64.lld",
      stageBuiltinsRelativePath: "lib/clang/20/lib/darwin/libclang_rt.builtins.a",
      stageAdditionalRequiredRelativePaths: ["lib/libSystem.tbd"],
      stageEmptySDKRelativePath: "empty-sdk",
      stageStubLibraryRelativeDirectory: "lib",
      stubLibraryLinkArguments: ["-lSystem"],
      undefinedSymbolArguments: ["-Wl,-undefined,error", "-Wl,-fatal_warnings"])

  #elseif os(Linux) && arch(x86_64)

    static let hostProfile = HostProfile(
      identifier: "linux-x86_64",
      target: "x86_64-unknown-linux-gnu",
      // `-march=x86-64-v3` (AVX2 + FMA + BMI2; Haswell 2013 / Zen 1 and
      // later) is a NUMERICAL FLOOR, not a performance preference. Do not
      // relax it to x86-64 or x86-64-v2 to widen hardware support.
      //
      // toolchain/include/dgen_simd_compat.h implements the NEON intrinsics
      // that CRenderer emits unconditionally; `vfmaq_f32` / `vfmsq_f32` are
      // implemented with `__builtin_elementwise_fma`, whose contract is a
      // SINGLE-ROUNDED fused multiply-add — that is what the ARM original
      // does, and the transcendental polynomials in dgen_runtime.h are
      // evaluated assuming it. Measured on clang 22 with the exact
      // optimization flags below (-O3 -ffast-math -ffp-contract=fast):
      //
      //   -march=x86-64     -> mulps  + addps      (UNFUSED, double-rounds)
      //   -march=x86-64-v2  -> mulps  + addps      (UNFUSED, double-rounds)
      //   -march=x86-64-v3  -> vfmadd231ps         (fused, single-rounded)
      //
      // Below v3 there is no FMA instruction to select, so every FMA in the
      // shim silently degrades to a double-rounded mul+add and the x86 build
      // stops being numerically faithful to the arm64 one. Supporting
      // pre-Haswell hardware would require a soft-FMA fallback in the shim,
      // not a weaker -march here.
      cpuArguments: ["-march=x86-64-v3"],
      // GNU ld only strips at section granularity, so per-function/-datum
      // sections are the compile-side half of `--gc-sections`. Darwin's
      // `-dead_strip` needs no equivalent: Mach-O atomizes by symbol.
      sectionArguments: ["-ffunction-sections", "-fdata-sections"],
      sharedLibraryArgument: "-shared",
      deadStripArgument: "-Wl,--gc-sections",
      artifactExtension: "so",
      supportsStagedToolchain: true,
      // Revision 2: the ELF staged-toolchain path. The preamble is a cache
      // key, so a profile whose emitted arguments change must bump it or
      // artifacts compiled under revision 1's system-clang path would be
      // reused for hermetic ones.
      signaturePreamble: ["host-profile=linux-x86_64", "host-profile-revision=2"],
      installNameArgument: { base in "-Wl,-soname,\(base)" },
      stageLinkerRelativePath: "bin/ld.lld",
      stageBuiltinsRelativePath:
        "lib/clang/20/lib/x86_64-unknown-linux-gnu/libclang_rt.builtins.a",
      // The ELF stage carries no stub library and no SDK to neutralise, so it
      // preflights nothing beyond the common set.
      stageAdditionalRequiredRelativePaths: [],
      stageEmptySDKRelativePath: nil,
      stageStubLibraryRelativeDirectory: nil,
      stubLibraryLinkArguments: [],
      undefinedSymbolArguments: [])

  #else

    #error(
      """
      DGen has no compile/link policy for this host. Supported hosts are \
      macOS/arm64 and Linux/x86_64. Add a HostProfile in \
      Sources/DGen/DGenToolchainPolicy.swift before building here.
      """)

  #endif

  /// Clang target triple for the selected host profile.
  public static var target: String { hostProfile.target }

  /// Extension (no leading dot) of the shared object this policy produces:
  /// `dylib` on Darwin, `so` on Linux. Callers that name the user-visible
  /// artifact must use this rather than hardcoding either.
  public static var artifactExtension: String { hostProfile.artifactExtension }

  // MARK: - Host-independent contract

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

  private static let sharedContractArguments = [
    "-flto=thin",
    "-fPIC",
    "-fvisibility=hidden",
    "-fno-stack-protector",
    "-fno-asynchronous-unwind-tables",
  ]

  /// Full contract argument list for the selected host. Ordering matters:
  /// `-x c` stays last so it applies to the source file, which every caller
  /// appends after these.
  private static var contractArguments: [String] {
    hostProfile.cpuArguments
      + sharedContractArguments
      + hostProfile.sectionArguments
      + ["-std=c11", "-x", "c"]
  }

  /// Directory containing the running DGenLisp executable.
  ///
  /// Published binaries must never derive runtime resources from `#filePath`:
  /// that value names the checkout used to build the executable. Resources
  /// distributed with DGenLisp live beside the executable instead.
  public static var executableDirectory: URL {
    if let executable = Bundle.main.executableURL {
      return executable.deletingLastPathComponent()
    }
    return URL(fileURLWithPath: CommandLine.arguments[0])
      .standardizedFileURL
      .deletingLastPathComponent()
  }

  static func resolveDevelopmentRuntimeInclude(
    environment: [String: String],
    executableDirectory: URL
  ) -> URL {
    if let override = environment["DGEN_RUNTIME_INCLUDE"], !override.isEmpty {
      return URL(fileURLWithPath: override)
    }
    return executableDirectory.appendingPathComponent("toolchain/include", isDirectory: true)
  }

  public static var developmentRuntimeInclude: URL {
    resolveDevelopmentRuntimeInclude(
      environment: ProcessInfo.processInfo.environment,
      executableDirectory: executableDirectory)
  }

  // MARK: - Stage-root resolution

  /// Where a resolved stage root came from, so the "no staged toolchain on
  /// this host" error can name the knob the caller actually turned.
  private enum StageRootOrigin: String {
    case explicit = "--toolchain-root"
    case environment = "DGEN_TOOLCHAIN_STAGE_ROOT"
  }

  private static func resolvedStageRootWithOrigin(
    explicit: String?
  ) -> (root: URL, origin: StageRootOrigin)? {
    if let explicit, !explicit.isEmpty {
      return (URL(fileURLWithPath: explicit, isDirectory: true), .explicit)
    }
    if let stagePath = ProcessInfo.processInfo.environment["DGEN_TOOLCHAIN_STAGE_ROOT"],
      !stagePath.isEmpty
    {
      return (URL(fileURLWithPath: stagePath, isDirectory: true), .environment)
    }
    return nil
  }

  /// Resolves the staged toolchain root for one invocation.
  ///
  /// An explicit root — the host-selected `--toolchain-root` — always wins over
  /// the `DGEN_TOOLCHAIN_STAGE_ROOT` development fallback. `nil` means no
  /// staged toolchain was selected at all, which is the only case that may use
  /// the system-Clang development path.
  public static func resolvedStageRoot(explicit: String? = nil) -> URL? {
    resolvedStageRootWithOrigin(explicit: explicit)?.root
  }

  // MARK: - Invocations

  public static func compileInvocation(
    outputPath: String,
    sourcePath: String,
    toolchainRoot: String? = nil
  ) throws -> DGenCompilerInvocation {
    // A selected root is binding: an incomplete stage is an error, never a
    // silent downgrade to the system compiler. On a host with no staged
    // toolchain at all, selecting one is the same class of error.
    if let selection = resolvedStageRootWithOrigin(explicit: toolchainRoot) {
      guard hostProfile.supportsStagedToolchain else {
        throw NSError(
          domain: "DGenToolchainPolicy",
          code: 2,
          userInfo: [
            NSLocalizedDescriptionKey: """
              No staged DGen toolchain exists for this platform \
              (\(hostProfile.identifier)), but one was selected via \
              \(selection.origin.rawValue): \(selection.root.path)
              Stages are target-specific and no distribution is published \
              for this host. Remove \(selection.origin.rawValue) to use the \
              system-Clang development path; DGen will not silently \
              downgrade to it while a stage root is selected.
              """
          ])
      }
      try verifyHostSupportsBaselineISA()
      return try embeddedInvocation(
        stageRoot: selection.root,
        outputPath: outputPath,
        sourcePath: sourcePath)
    }
    try verifyHostSupportsBaselineISA()
    let runtimeInclude = developmentRuntimeInclude
    let runtimeHeader = runtimeInclude.appendingPathComponent("dgen_runtime.h")
    guard FileManager.default.isReadableFile(atPath: runtimeHeader.path) else {
      throw NSError(
        domain: "DGenToolchainPolicy",
        code: 4,
        userInfo: [
          NSLocalizedDescriptionKey: """
            DGen runtime headers are unavailable. Resolved dgen_runtime.h \
            path: \(runtimeHeader.path)
            Set DGEN_RUNTIME_INCLUDE to the directory containing \
            dgen_runtime.h, or install toolchain/include beside DGenLisp.
            """
        ])
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

  /// Fails the compile up front when the machine cannot execute the ISA
  /// baseline the policy is about to compile for.
  ///
  /// Emitting `-march=x86-64-v3` code onto a pre-AVX2 machine produces an
  /// artifact that builds and links cleanly and then dies with SIGILL inside
  /// `dgen_process_v1` — on the audio thread, with no diagnostic. A cheap
  /// `/proc/cpuinfo` check turns that into a compile-time error naming the
  /// missing feature. Set `DGEN_ALLOW_UNSUPPORTED_HOST_CPU=1` when
  /// deliberately building on one machine for another; the flags do not
  /// change, only this guard is lifted.
  public static func verifyHostSupportsBaselineISA() throws {
    #if os(Linux) && arch(x86_64)
      if let bypass = ProcessInfo.processInfo.environment["DGEN_ALLOW_UNSUPPORTED_HOST_CPU"],
        bypass == "1"
      {
        return
      }
      guard let missing = missingBaselineCPUFeatures, !missing.isEmpty else { return }
      throw NSError(
        domain: "DGenToolchainPolicy",
        code: 3,
        userInfo: [
          NSLocalizedDescriptionKey: """
            This CPU does not support the x86-64-v3 baseline DGen compiles \
            for. Missing feature(s): \(missing.joined(separator: ", ")).
            x86-64-v3 (AVX2 + FMA, Haswell/Zen 1 and later) is required for \
            single-rounded FMA in toolchain/include/dgen_simd_compat.h; \
            below it every fused multiply-add silently double-rounds. \
            Building anyway would produce a shared object that loads and then \
            traps with SIGILL on the audio thread. Set \
            DGEN_ALLOW_UNSUPPORTED_HOST_CPU=1 if you are deliberately \
            building for a different machine.
            """
        ])
    #endif
  }

  #if os(Linux) && arch(x86_64)
    /// `nil` when the feature list could not be read at all — an unreadable
    /// `/proc` is not evidence of an unsupported CPU, so the guard stands down.
    private static let missingBaselineCPUFeatures: [String]? = {
      guard let info = try? String(contentsOfFile: "/proc/cpuinfo", encoding: .utf8) else {
        return nil
      }
      guard
        let flagsLine = info.split(separator: "\n").first(where: { $0.hasPrefix("flags") })
      else { return nil }
      let flags = Set(flagsLine.split(whereSeparator: { $0 == " " || $0 == "\t" }).map(String.init))
      // The v3 level is AVX2+FMA+BMI1/2+MOVBE+LZCNT+F16C. AVX2 and FMA are the
      // two that carry the numerical contract and the two whose absence
      // guarantees a SIGILL in generated code; the rest ride along on any CPU
      // that has these.
      return ["avx2", "fma"].filter { !flags.contains($0) }
    }()
  #endif

  public static func systemDevelopmentInvocation(
    outputPath: String,
    sourcePath: String
  ) -> DGenCompilerInvocation {
    let artifactName = URL(fileURLWithPath: outputPath).lastPathComponent
    let arguments =
      ["-target", target]
      + optimizationArguments
      + contractArguments
      + [
        "-I", developmentRuntimeInclude.path,
        hostProfile.sharedLibraryArgument,
        hostProfile.deadStripArgument,
        hostProfile.installNameArgument(artifactName),
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
    guard hostProfile.supportsStagedToolchain else {
      throw NSError(
        domain: "DGenToolchainPolicy",
        code: 2,
        userInfo: [
          NSLocalizedDescriptionKey: """
            No staged DGen toolchain exists for this platform \
            (\(hostProfile.identifier)); stages are target-specific and none \
            is published for this host. \
            Requested stage root: \(stageRoot.path)
            """
        ])
    }

    let clang = stageRoot.appendingPathComponent("bin/dgen-clang").path
    let linker = stageRoot.appendingPathComponent(hostProfile.stageLinkerRelativePath).path
    let resourceDirectory = stageRoot.appendingPathComponent("lib/clang/20").path
    let builtins = stageRoot.appendingPathComponent(hostProfile.stageBuiltinsRelativePath).path
    let include = stageRoot.appendingPathComponent("include").path
    let emptySDK = hostProfile.stageEmptySDKRelativePath.map {
      stageRoot.appendingPathComponent($0).path
    }
    let stubLibrarySearchArguments = hostProfile.stageStubLibraryRelativeDirectory.map {
      ["-L\(stageRoot.appendingPathComponent($0).path)"]
    } ?? []
    let sysrootArguments = emptySDK.map { ["-isysroot", $0] } ?? []

    // Every file the staged layout promises (toolchain/LAYOUT.md). A selected
    // root that cannot satisfy it fails the compile outright.
    let missing = ([
      clang, linker, builtins,
      stageRoot.appendingPathComponent("include/dgen_runtime.h").path,
    ]
      + hostProfile.stageAdditionalRequiredRelativePaths.map {
        stageRoot.appendingPathComponent($0).path
      }
      + [stageRoot.appendingPathComponent("VERSION.json").path])
      .filter { !FileManager.default.fileExists(atPath: $0) }

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

    if let emptySDK {
      try FileManager.default.createDirectory(
        atPath: emptySDK,
        withIntermediateDirectories: true)
    }

    var arguments = ["-target", target]
    arguments += optimizationArguments
    arguments += contractArguments
    arguments += ["-ffreestanding", "-nostdinc"]
    arguments += sysrootArguments
    arguments += [
      "-resource-dir", resourceDirectory,
      "-isystem", "\(resourceDirectory)/include",
      "-I", include,
      "-fuse-ld=\(linker)",
      "-nostdlib",
      hostProfile.sharedLibraryArgument,
      hostProfile.installNameArgument(URL(fileURLWithPath: outputPath).lastPathComponent),
    ]
    arguments += hostProfile.undefinedSymbolArguments
    arguments.append(hostProfile.deadStripArgument)
    arguments += stubLibrarySearchArguments
    arguments += [sourcePath, "-x", "none", builtins]
    arguments += hostProfile.stubLibraryLinkArguments
    arguments += ["-o", outputPath]
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
    (hostProfile.signaturePreamble + [
      "dgen-toolchain-policy=\(policyVersion)",
      "mode=\(mode)",
      "executable=\(executable)",
    ] + arguments).joined(separator: "\n")
  }
}
