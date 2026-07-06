import Foundation

struct RecoveryRow: Codable {
  var name: String
  var unit: String
  var trueValue: Float
  var recoveredValue: Float
  var absoluteError: Float
  var relativeError: Float
  var tolerance: Float
  var pass: Bool
}

struct SynthIDReport: Codable {
  var createdAtUTC: String
  var rung: Int
  var pass: Bool
  var initLoss: Float
  var finalLoss: Float
  var lossRatio: Float
  var rows: [RecoveryRow]
}

enum ReportWriter {
  static func make(
    rung: Int,
    trueParams: PatchValues?,
    recovered: PatchValues,
    initLoss: Float,
    finalLoss: Float
  ) -> SynthIDReport {
    let rows: [RecoveryRow]
    if let trueParams {
      rows = KickParamSpecs.all.map { spec in
        let expected = trueParams[spec.name]
        let actual = recovered[spec.name]
        let absErr = abs(actual - expected)
        let relErr = absErr / max(abs(expected), 1e-9)
        return RecoveryRow(
          name: spec.name,
          unit: spec.unit,
          trueValue: expected,
          recoveredValue: actual,
          absoluteError: absErr,
          relativeError: relErr,
          tolerance: spec.tolerance,
          pass: relErr <= spec.tolerance)
      }
    } else {
      rows = []
    }

    let lossRatio = finalLoss / max(initLoss, 1e-12)
    let parameterPass = rows.isEmpty || rows.allSatisfy(\.pass)
    let lossPass = rows.isEmpty || lossRatio <= 0.02
    return SynthIDReport(
      createdAtUTC: timestampUTC(),
      rung: rung,
      pass: parameterPass && lossPass,
      initLoss: initLoss,
      finalLoss: finalLoss,
      lossRatio: lossRatio,
      rows: rows)
  }

  static func write(report: SynthIDReport, to outDir: URL) throws {
    try writeJSON(report, to: outDir.appendingPathComponent("report.json"))
    try markdown(report: report).write(
      to: outDir.appendingPathComponent("report.md"),
      atomically: true,
      encoding: .utf8)
  }

  static func markdown(report: SynthIDReport) -> String {
    var text = "# SynthID Report\n\n"
    text += "- Rung: \(report.rung)\n"
    text += "- Pass: \(report.pass ? "yes" : "no")\n"
    text += "- Init loss: \(String(format: "%.6f", report.initLoss))\n"
    text += "- Final loss: \(String(format: "%.6f", report.finalLoss))\n"
    text += "- Loss ratio: \(String(format: "%.6f", report.lossRatio))\n\n"
    guard !report.rows.isEmpty else { return text }
    text += "| Parameter | True | Recovered | Abs err | Rel err | Tol | Pass |\n"
    text += "| --- | ---: | ---: | ---: | ---: | ---: | --- |\n"
    for row in report.rows {
      text +=
        "| \(row.name) | \(fmt(row.trueValue)) | \(fmt(row.recoveredValue)) | \(fmt(row.absoluteError)) | \(pct(row.relativeError)) | \(pct(row.tolerance)) | \(row.pass ? "yes" : "no") |\n"
    }
    return text
  }

  private static func fmt(_ value: Float) -> String {
    String(format: "%.6g", value)
  }

  private static func pct(_ value: Float) -> String {
    String(format: "%.2f%%", value * 100.0)
  }
}
