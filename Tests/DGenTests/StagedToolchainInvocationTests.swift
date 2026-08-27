import Foundation
import XCTest

@testable import DGen

/// Locks the staged (hermetic) compile invocation per host profile.
///
/// The emitted argument list is a compilation-cache key, so it is not free to
/// drift: an argument added, dropped, or reordered must be a deliberate change
/// with a `signaturePreamble` revision bump behind it. The Mach-O expectation
/// below is the pre-ELF-port list verbatim, which is what makes it a
/// regression test rather than a restatement of the implementation.
final class StagedToolchainInvocationTests: XCTestCase {
  /// A stage root whose every preflighted file exists, so the invocation
  /// builder is exercised rather than the incompleteness guard.
  private func makeCompleteStageRoot() throws -> URL {
    let root = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
      .appendingPathComponent("dgen-stage-\(UUID().uuidString)", isDirectory: true)
    let profile = DGenToolchainPolicy.hostProfile
    let relativePaths =
      [
        "bin/dgen-clang",
        profile.stageLinkerRelativePath,
        profile.stageBuiltinsRelativePath,
        "include/dgen_runtime.h",
        "VERSION.json",
      ] + profile.stageAdditionalRequiredRelativePaths
    for relative in relativePaths {
      let file = root.appendingPathComponent(relative)
      try FileManager.default.createDirectory(
        at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
      try Data().write(to: file)
    }
    addTeardownBlock { try? FileManager.default.removeItem(at: root) }
    return root
  }

  func testStagedInvocationUsesTheStagedCompilerAndNeverASystemOne() throws {
    let root = try makeCompleteStageRoot()

    let invocation = try DGenToolchainPolicy.embeddedInvocation(
      stageRoot: root, outputPath: "/out/patch.bin", sourcePath: "/src/patch.c")

    XCTAssertEqual(invocation.executable, root.appendingPathComponent("bin/dgen-clang").path)
    XCTAssertFalse(invocation.executable.hasPrefix("/usr/"))
    // -fuse-ld= must name the staged linker, not let the driver search PATH.
    XCTAssertTrue(
      invocation.arguments.contains(
        "-fuse-ld=\(root.appendingPathComponent(DGenToolchainPolicy.hostProfile.stageLinkerRelativePath).path)"
      ))
  }

  func testIncompleteStageRootIsRejectedBeforeAnyCompilerRuns() throws {
    let root = try makeCompleteStageRoot()
    let linker = root.appendingPathComponent(
      DGenToolchainPolicy.hostProfile.stageLinkerRelativePath)
    try FileManager.default.removeItem(at: linker)

    XCTAssertThrowsError(
      try DGenToolchainPolicy.embeddedInvocation(
        stageRoot: root, outputPath: "/out/patch.bin", sourcePath: "/src/patch.c")
    ) { error in
      // The diagnostic must name the missing file: an incomplete stage is a
      // packaging bug, and "incomplete" alone does not say which one.
      XCTAssertTrue("\(error)".contains(linker.lastPathComponent), "\(error)")
    }
  }

  #if os(Linux) && arch(x86_64)

    /// The ELF link deliberately leaves the libc/libm surface undefined for the
    /// loader to bind at `dlopen`; that surface is what
    /// `abi/libsystem-symbols-v1-elf.txt` allowlists and the ELF audit checks.
    /// Mach-O's strictness flags would reject every valid artifact here, and
    /// its libSystem stub does not exist on this platform at all.
    func testELFInvocationCarriesNoMachOIsmsAndDoesNotDemandAResolvedLink() throws {
      let root = try makeCompleteStageRoot()

      let arguments = try DGenToolchainPolicy.embeddedInvocation(
        stageRoot: root, outputPath: "/out/patch.so", sourcePath: "/src/patch.c"
      ).arguments

      for machOism in [
        "-isysroot", "-lSystem", "-Wl,-undefined,error", "-Wl,-fatal_warnings",
      ] {
        XCTAssertFalse(arguments.contains(machOism), "unexpected Mach-O argument: \(machOism)")
      }
      XCTAssertFalse(
        arguments.contains { $0.hasSuffix("libSystem.tbd") || $0.hasPrefix("-L") },
        "the ELF stage links against no stub library")
      XCTAssertTrue(arguments.contains("-shared"))
      XCTAssertTrue(arguments.contains("-nostdlib"))
      XCTAssertTrue(arguments.contains("-Wl,-soname,patch.so"))
      XCTAssertTrue(arguments.contains("-Wl,--gc-sections"))
      XCTAssertTrue(arguments.contains("-march=x86-64-v3"))
    }

  #endif

  #if os(macOS) && arch(arm64)

    /// The Mach-O argument list verbatim as it stood before host profiles grew
    /// staged-layout fields. Artifact caches keyed on this list stay valid only
    /// while it is byte-identical.
    func testMachOInvocationIsUnchangedByTheELFPort() throws {
      let root = try makeCompleteStageRoot()
      let resourceDirectory = root.appendingPathComponent("lib/clang/20").path

      let arguments = try DGenToolchainPolicy.embeddedInvocation(
        stageRoot: root, outputPath: "/out/patch.dylib", sourcePath: "/src/patch.c"
      ).arguments

      let tail = [
        "-ffreestanding",
        "-nostdinc",
        "-isysroot", root.appendingPathComponent("empty-sdk").path,
        "-resource-dir", resourceDirectory,
        "-isystem", "\(resourceDirectory)/include",
        "-I", root.appendingPathComponent("include").path,
        "-fuse-ld=\(root.appendingPathComponent("bin/ld64.lld").path)",
        "-nostdlib",
        "-dynamiclib",
        "-Wl,-install_name,@rpath/patch.dylib",
        "-Wl,-undefined,error",
        "-Wl,-fatal_warnings",
        "-Wl,-dead_strip",
        "-L\(root.appendingPathComponent("lib").path)",
        "/src/patch.c",
        "-x", "none",
        root.appendingPathComponent("lib/clang/20/lib/darwin/libclang_rt.builtins.a").path,
        "-lSystem",
        "-o", "/out/patch.dylib",
      ]
      XCTAssertEqual(Array(arguments.suffix(tail.count)), tail)
    }

  #endif
}
