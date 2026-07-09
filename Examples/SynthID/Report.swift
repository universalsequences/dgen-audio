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

struct EquivalenceRow: Codable {
  var name: String
  var trueValue: Float
  var recoveredValue: Float
  var relativeError: Float
}

struct SynthIDReport: Codable {
  var createdAtUTC: String
  var rung: Int
  var pass: Bool
  var initLoss: Float
  var finalLoss: Float
  var lossRatio: Float
  var rows: [RecoveryRow]
  var equivalences: [EquivalenceRow]
}

enum ReportWriter {
  static func make(
    rung: Int,
    trueParams: PatchValues?,
    recovered: PatchValues,
    initLoss: Float,
    finalLoss: Float,
    includeNoiseCutoff: Bool = true
  ) -> SynthIDReport {
    // The output tanh((bodyAmp·body + clickAmp·click + noiseAmp·noise)·drive)·outGain
    // depends only on the products amp·drive and on outGain: any parameter set with
    // the same products renders bit-identical audio, so the individual factors are
    // not identifiable from the target and are reported unscored. The scored rows
    // for the degenerate directions are the products, at the factor's tolerance.
    let degenerate: Set<String> = ["bodyAmp", "clickAmp", "noiseAmp", "drive"]
    var rows: [RecoveryRow] = []
    var equivalences: [EquivalenceRow] = []
    if let trueParams {
      rows = KickParamSpecs.all.filter { spec in
        (includeNoiseCutoff || spec.name != "noiseCutoff") && !degenerate.contains(spec.name)
      }.map { spec in
        makeRow(
          name: spec.name,
          unit: spec.unit,
          trueValue: trueParams[spec.name],
          recoveredValue: recovered[spec.name],
          tolerance: spec.tolerance)
      }
      for (factor, tolerance) in [("bodyAmp", Float(0.10)), ("clickAmp", Float(0.20)), ("noiseAmp", Float(0.20))] {
        rows.append(
          makeRow(
            name: "\(factor)*drive",
            unit: "lin",
            trueValue: trueParams[factor] * trueParams.drive,
            recoveredValue: recovered[factor] * recovered.drive,
            tolerance: tolerance))
      }
      equivalences = makeEquivalences(trueParams: trueParams, recovered: recovered)
      for name in ["bodyAmp", "clickAmp", "noiseAmp", "drive"] {
        equivalences.append(
          makeEquivalence(
            name: "\(name) (unscored factor)",
            trueValue: trueParams[name],
            recoveredValue: recovered[name]))
      }
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
      rows: rows,
      equivalences: equivalences)
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
    text +=
      "Note: `tanh((bodyAmp·body + clickAmp·click + noiseAmp·noise)·drive)·outGain` "
      + "depends only on the products `amp·drive` and `outGain`; parameter sets with equal "
      + "products render identical audio. The products are scored; the factors are listed "
      + "unscored under Effective Gain Products.\n\n"
    text += "| Parameter | True | Recovered | Abs err | Rel err | Tol | Pass |\n"
    text += "| --- | ---: | ---: | ---: | ---: | ---: | --- |\n"
    for row in report.rows {
      text +=
        "| \(row.name) | \(fmt(row.trueValue)) | \(fmt(row.recoveredValue)) | \(fmt(row.absoluteError)) | \(pct(row.relativeError)) | \(pct(row.tolerance)) | \(row.pass ? "yes" : "no") |\n"
    }
    if !report.equivalences.isEmpty {
      text += "\n## Effective Gain Products\n\n"
      text += "| Product | True | Recovered | Rel err |\n"
      text += "| --- | ---: | ---: | ---: |\n"
      for row in report.equivalences {
        text +=
          "| \(row.name) | \(fmt(row.trueValue)) | \(fmt(row.recoveredValue)) | \(pct(row.relativeError)) |\n"
      }
    }
    return text
  }

  private static func makeRow(
    name: String,
    unit: String,
    trueValue: Float,
    recoveredValue: Float,
    tolerance: Float
  ) -> RecoveryRow {
    let absErr = abs(recoveredValue - trueValue)
    let relErr = absErr / max(abs(trueValue), 1e-9)
    return RecoveryRow(
      name: name,
      unit: unit,
      trueValue: trueValue,
      recoveredValue: recoveredValue,
      absoluteError: absErr,
      relativeError: relErr,
      tolerance: tolerance,
      pass: relErr <= tolerance)
  }

  private static func makeEquivalences(
    trueParams: PatchValues,
    recovered: PatchValues
  ) -> [EquivalenceRow] {
    [
      makeEquivalence(
        name: "bodyAmp*drive*outGain",
        trueValue: trueParams.bodyAmp * trueParams.drive * trueParams.outGain,
        recoveredValue: recovered.bodyAmp * recovered.drive * recovered.outGain),
      makeEquivalence(
        name: "clickAmp*drive*outGain",
        trueValue: trueParams.clickAmp * trueParams.drive * trueParams.outGain,
        recoveredValue: recovered.clickAmp * recovered.drive * recovered.outGain),
      makeEquivalence(
        name: "noiseAmp*drive*outGain",
        trueValue: trueParams.noiseAmp * trueParams.drive * trueParams.outGain,
        recoveredValue: recovered.noiseAmp * recovered.drive * recovered.outGain),
    ]
  }

  private static func makeEquivalence(
    name: String,
    trueValue: Float,
    recoveredValue: Float
  ) -> EquivalenceRow {
    let relErr = abs(recoveredValue - trueValue) / max(abs(trueValue), 1e-9)
    return EquivalenceRow(
      name: name,
      trueValue: trueValue,
      recoveredValue: recoveredValue,
      relativeError: relErr)
  }

  private static func fmt(_ value: Float) -> String {
    String(format: "%.6g", value)
  }

  private static func pct(_ value: Float) -> String {
    String(format: "%.2f%%", value * 100.0)
  }
}
