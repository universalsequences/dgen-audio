import Foundation

enum UIEnvelopeRole: String, CaseIterable, Codable {
  case attack
  case decay
  case sustain
  case release
}

func parseUIMetadataSymbol(_ value: String?, attribute: String, paramName: String) throws -> String? {
  guard let value else { return nil }
  guard !value.isEmpty, !value.hasPrefix("@") else {
    throw LispError.validationError("param '\(paramName)' has \(attribute) without a symbol value")
  }
  return value
}

func parseUIEnvelopeRole(_ value: String?, paramName: String) throws -> UIEnvelopeRole? {
  guard let value else { return nil }
  guard let role = UIEnvelopeRole(rawValue: value.lowercased()) else {
    let allowed = UIEnvelopeRole.allCases.map(\.rawValue).joined(separator: ", ")
    throw LispError.validationError(
      "param '\(paramName)' has invalid @role '\(value)'; expected one of: \(allowed)")
  }
  return role
}

func validateParamUIMetadata(_ params: [ParamInfo]) throws {
  var envelopes: [String: (group: String?, roles: [UIEnvelopeRole: String])] = [:]

  for param in params {
    if param.role != nil && param.env == nil {
      throw LispError.validationError("param '\(param.name)' has @role but is missing @env")
    }
    if let env = param.env, param.role == nil {
      throw LispError.validationError("param '\(param.name)' has @env '\(env)' but is missing @role")
    }

    guard let env = param.env else { continue }

    var envelope = envelopes[env] ?? (group: nil, roles: [:])
    if let group = param.group {
      if let existingGroup = envelope.group, existingGroup != group {
        throw LispError.validationError(
          "envelope '\(env)' has conflicting @group values '\(existingGroup)' and '\(group)'")
      }
      envelope.group = group
    }

    if let role = param.role {
      if let existingParam = envelope.roles[role] {
        throw LispError.validationError(
          "envelope '\(env)' has duplicate @role \(role.rawValue) on params '\(existingParam)' and '\(param.name)'")
      }
      envelope.roles[role] = param.canonicalName
    }

    envelopes[env] = envelope
  }
}

// MARK: - @options

/// Label source for a discrete-choice param declared with `@options`.
///
/// Inline labels are baked at compile time; the asset form keeps the
/// reference so the host re-resolves labels from the tensor's file (and
/// tracks `tensor-param` pushes) instead of freezing them here.
enum ParamOptions {
  case labels([String])
  case asset(tensor: String, file: String, key: String)
}

/// Metadata key holding the labels for an asset-backed `@options` param.
let defaultParamOptionsKey = "sets"

/// Normalizes an authored `@options-key` to the asset JSON key spelling
/// (`wave-names` → `wave_names`), matching the metadata keys in the
/// tensor asset format.
func normalizeParamOptionsKey(_ raw: String) -> String {
  raw.trimmingCharacters(in: CharacterSet(charactersIn: "\"'"))
    .replacingOccurrences(of: "-", with: "_")
}

/// Parses an inline `@options ["a" "b c"]` bracket list into labels.
/// Bare (unquoted) tokens are accepted so enum-ish params can be written
/// `@options [lowpass highpass]`; whitespace and commas separate them.
func parseInlineParamOptionLabels(_ raw: String, paramName: String) throws -> [String] {
  var body = raw.trimmingCharacters(in: .whitespaces)
  guard body.hasPrefix("["), body.hasSuffix("]") else {
    throw LispError.validationError(
      "param '\(paramName)' has a malformed @options list: expected [\"a\" \"b\"]")
  }
  body.removeFirst()
  body.removeLast()

  var labels: [String] = []
  var current = ""
  var inQuotes = false

  func flush() {
    if !current.isEmpty {
      labels.append(current)
      current = ""
    }
  }

  for char in body {
    if char == "\"" {
      if inQuotes {
        labels.append(current)
        current = ""
        inQuotes = false
      } else {
        flush()
        inQuotes = true
      }
      continue
    }
    if inQuotes {
      current.append(char)
      continue
    }
    if char.isWhitespace || char == "," {
      flush()
      continue
    }
    current.append(char)
  }
  guard !inQuotes else {
    throw LispError.validationError(
      "param '\(paramName)' has an unterminated string in its @options list")
  }
  flush()

  guard !labels.isEmpty else {
    throw LispError.validationError("param '\(paramName)' has an empty @options list")
  }
  return labels
}

/// Param attributes the compiler understands. Everything else authored on a
/// `(param …)` form is carried through to the manifest verbatim as an inert
/// UI hint rather than being silently dropped.
let knownParamAttributes: Set<String> = [
  "@default", "@min", "@max", "@unit", "@hidden", "@group", "@env", "@role",
  "@generated", "@generated-for", "@modulator-slot", "@options", "@options-key",
  "@name",
]

/// Collects the authored attributes the compiler does not consume itself.
/// `@mod*` attributes belong to modulation lowering and stay out of the map.
func inertParamAttributes(_ attributes: [(name: String, value: String)]) -> [String: String] {
  var extras: [String: String] = [:]
  for (name, value) in attributes {
    if knownParamAttributes.contains(name) || name.hasPrefix("@mod") { continue }
    extras[String(name.dropFirst())] =
      value.trimmingCharacters(in: CharacterSet(charactersIn: "\""))
  }
  return extras
}
