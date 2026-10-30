import Foundation
import XCTest

@testable import DGenAnalysis

/// Diagnostics only (DGEN_ANALYSIS_DEBUG=1): prints what the analysis
/// recovers on the synthetic kicks.
final class AnalysisDebugTests: XCTestCase {
    func testPrintSyntheticRecovery() throws {
        try XCTSkipIf(ProcessInfo.processInfo.environment["DGEN_ANALYSIS_DEBUG"] == nil)
        for (name, p, secs) in [("noisy", Synth.kick(noise: true, E: 0.3), 0.3),
                                ("second", Synth.kick(noise: false, second: true), 0.4)] {
            var dbg = KickDebug()
            let (a, prep) = KickAnalyzer.analyze(name: name, audio: Synth.audio(p, seconds: secs), debug: &dbg)
            print("== \(name) crop \(prep.cropStart) onset \(a.prep.onsetSeconds) drop \(a.drop)")
            for m in a.modes {
                print(String(format: "  F %8.2f D %6.1fms A %.3f G %.2f T %.1fms G2 %.2f T2 %.1fms %@ %@", m.F, m.D * 1000, m.A,
                             m.G, m.T * 1000, m.G2, m.T2 * 1000, m.region, String(describing: m.dropTestRatio)))
            }
        }
    }
}
