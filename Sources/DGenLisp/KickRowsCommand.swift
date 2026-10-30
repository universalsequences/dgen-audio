// `dgenlisp kick-rows`: slot correspondence (DGenAnalysis.KickSlots) from an analyze-kicks
// result to the voice's row layout, ready for `train-dataset --rows`.
//
//   dgenlisp kick-rows --analysis analysis.json --slots 8 -o rows.json [--template-out t.json]
//   dgenlisp kick-rows --layout v3 --slots 12 --analysis analysis.json -o rows_v3.json   (voice3.lisp)
//   dgenlisp kick-rows --layout snare1 --slots 24 --analysis analysis.json -o rows.json  (snare1.lisp; also `drum-rows`)

import DGenAnalysis
import Foundation

enum KickRowsCommand {
  static func run(arguments: [String]) -> Never {
    var analysisPath: String?, output: String?, templateOut: String?
    var slots = 8
    var energyWeighted = false
    var layout = "v2"
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
      case "--layout": layout = value()
      default: fputs("unknown kick-rows option \(a)\n", stderr); exit(2)
      }
      i += 1
    }
    guard let analysisPath, let output else {
      fputs("Usage: dgenlisp kick-rows [--layout v2|v3|snare1] --analysis analysis.json --slots K -o rows.json\n", stderr)
      exit(2)
    }
    do {
      let data = try Data(contentsOf: URL(fileURLWithPath: analysisPath))
      let ds = try JSONDecoder().decode(DatasetAnalysis.self, from: data)
      guard ["v2", "v3", "snare1"].contains(layout) else {
        fputs("kick-rows: --layout must be v2, v3 or snare1\n", stderr)
        exit(2)
      }
      let snare = layout == "snare1"
      let v3 = layout == "v3" || snare
      let template = KickSlotTemplate.estimate(ds.samples, slots: slots, energyWeighted: energyWeighted)
      let strikeTemplate = KickSlotTemplate.estimateStrikes(ds.samples, perRegion: 2)
      var rows: [String: Any] = [:]
      var report: [[String: Any]] = []
      for s in ds.samples {
        let a = v3 ? KickSlots.assignV3(s, template: template) : KickSlots.assign(s, template: template)
        if snare {
          rows[s.name] = SnareRows.row(s, slots: slots, template: template, assignment: a,
                                       strikeTemplate: strikeTemplate,
                                       strikeSlots: KickSlots.assignStrikes(s, template: strikeTemplate))
        } else if v3 {
          rows[s.name] = KickRowsV3.row(s, template: template, assignment: a, strikeTemplate: strikeTemplate,
                                        strikeSlots: KickSlots.assignStrikes(s, template: strikeTemplate))
        } else {
          rows[s.name] = KickRows.row(s, template: template, assignment: a)
        }
        let modal = s.modes.reduce(0) { $0 + $1.energy }
        var rep: [String: Any] = [
          "name": s.name, "filled": a.roleModes.compactMap { $0 }.count,
          "dropped": a.dropped.count, "dropped_energy_share": modal > 0 ? a.droppedEnergy / modal : 0,
          "dropped_modes": a.dropped.map { ["f": s.modes[$0].F, "a": s.modes[$0].A, "d": s.modes[$0].D] },
        ]
        if v3 {
          rep["strikes"] = (s.strikes ?? []).count
          rep["crackle"] = s.crackle?.present ?? false
          rep["events"] = (s.noise.events ?? []).count
        }
        report.append(rep)
      }
      var out: [String: Any] = ["layout": snare ? "snare" : "kick", "rows": rows,
                                "roles": template.roles.map { ["f": $0.frequency, "decay": $0.decay] },
                                "assignment": report]
      if v3 { out["strike_roles"] = strikeTemplate.roles.map { ["f": $0.frequency, "decay": $0.decay] } }
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
