import Foundation
import XCTest

@testable import DGen

final class PublishedResourceResolutionTests: XCTestCase {
  func testRuntimeHeadersDefaultRelativeToExecutableInsteadOfBuildCheckout() {
    let executableDirectory = URL(fileURLWithPath: "/opt/dgen/bin", isDirectory: true)

    let resolved = DGenToolchainPolicy.resolveDevelopmentRuntimeInclude(
      environment: [:],
      executableDirectory: executableDirectory)

    XCTAssertEqual(resolved.path, "/opt/dgen/bin/toolchain/include")
  }

  func testExecutableDirectoryResolvesDistributionSymlink() throws {
    let root = FileManager.default.temporaryDirectory
      .appendingPathComponent(UUID().uuidString, isDirectory: true)
    let distribution = root.appendingPathComponent("DGenLisp.dist", isDirectory: true)
    let executable = distribution.appendingPathComponent("DGenLisp")
    let symlink = root.appendingPathComponent("DGenLisp-current")
    defer { try? FileManager.default.removeItem(at: root) }

    try FileManager.default.createDirectory(at: distribution, withIntermediateDirectories: true)
    XCTAssertTrue(FileManager.default.createFile(atPath: executable.path, contents: Data()))
    try FileManager.default.createSymbolicLink(
      atPath: symlink.path, withDestinationPath: "DGenLisp.dist/DGenLisp")

    let resolved = DGenToolchainPolicy.resolveExecutableDirectory(
      executableURL: symlink,
      commandLineExecutable: "/unused")

    XCTAssertEqual(resolved.standardizedFileURL, distribution.standardizedFileURL)
  }

  func testRuntimeHeaderEnvironmentOverrideWins() {
    let resolved = DGenToolchainPolicy.resolveDevelopmentRuntimeInclude(
      environment: ["DGEN_RUNTIME_INCLUDE": "/srv/dgen/include"],
      executableDirectory: URL(fileURLWithPath: "/opt/dgen/bin", isDirectory: true))

    XCTAssertEqual(resolved.path, "/srv/dgen/include")
  }

  func testAuditScriptDefaultsRelativeToExecutableInsteadOfBuildCheckout() {
    let resolved = DGenBinaryAudit.executableRelativeScript(
      executableDirectory: URL(fileURLWithPath: "/opt/dgen/bin", isDirectory: true))

    #if canImport(Darwin)
      XCTAssertEqual(resolved, "/opt/dgen/bin/scripts/audit-dgen-dylib.sh")
    #else
      XCTAssertEqual(resolved, "/opt/dgen/bin/scripts/audit-dgen-elf-so.sh")
    #endif
  }
}
