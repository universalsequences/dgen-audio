// `dgenlisp kick-rows`: slot correspondence (DGenAnalysis.KickSlots) from an analyze-kicks
// result to the voice's row layout, ready for `train-dataset --rows`.
//
//   dgenlisp kick-rows --analysis analysis.json --slots 8 -o rows.json [--template-out t.json]

import DGenAnalysis
import Foundation

enum KickRowsCommand {
  static func run(arguments: [String]) -> Never {
    var analysisPath: String?, output: String?, templateOut: String?
    var slots = 8
    var energyWeighted = false
    var i = 0
    while i < arguments.count {
      let a = arguments[i]
      func value() -> String {
        i += 1
        guard i < arguments.count else { fputs("missing value for \(a)\n", stderr); exit(2) }
        return arguments[i]
      }
      switch a {
      case "--analysis": analysisPath = value()
      case "--slots": slots = Int(value()) ?? slots
      case "-o", "--output": output = value()
      case "--template-out": templateOut = value()
      case "--energy-weighted": energyWeighted = true
      default: fputs("unknown kick-rows option \(a)\n", stderr); exit(2)
      }
      i += 1
    }
    guard let analysisPath, let output else {
      fputs("Usage: dgenlisp kick-rows --analysis analysis.json --slots K -o rows.json\n", stderr)
      exit(2)
    }
    do {
      let data = try Data(contentsOf: URL(fileURLWithPath: analysisPath))
      let ds = try JSONDecoder().decode(DatasetAnalysis.self, from: data)
      let template = KickSlotTemplate.estimate(ds.samples, slots: slots, energyWeighted: energyWeighted)
      var rows: [String: Any] = [:]
      var report: [[String: Any]] = []
      for s in ds.samples {
        let a = KickSlots.assign(s, template: template)
        rows[s.name] = KickRows.row(s, template: template, assignment: a)
        let modal = s.modes.reduce(0) { $0 + $1.energy }
        report.append([
          "name": s.name, "filled": a.roleModes.compactMap { $0 }.count,
          "dropped": a.dropped.count, "dropped_energy_share": modal > 0 ? a.droppedEnergy / modal : 0,
        ])
      }
      let out: [String: Any] = ["layout": "kick", "rows": rows,
                                "roles": template.roles.map { ["f": $0.frequency, "decay": $0.decay] },
                                "assignment": report]
      try JSONSerialization.data(withJSONObject: out, options: [.prettyPrinted, .sortedKeys])
        .write(to: URL(fileURLWithPath: output))
      if let templateOut {
        try JSONEncoder().encode(template).write(to: URL(fileURLWithPath: templateOut))
      }
      let shares = report.map { $0["dropped_energy_share"] as! Double }.sorted()
      fputs(String(format: "kick-rows: %d rows, roles %@ Hz; dropped modal energy median %.4f max %.4f\n",
                   ds.samples.count, template.roles.map { String(format: "%.0f", $0.frequency) }.joined(separator: " "),
                   shares[shares.count / 2], shares.last ?? 0), stderr)
      exit(0)
    } catch {
      fputs("kick-rows error: \(error)\n", stderr)
      exit(1)
    }
  }
}
