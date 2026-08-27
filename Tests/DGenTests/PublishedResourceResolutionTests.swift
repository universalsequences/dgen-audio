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
