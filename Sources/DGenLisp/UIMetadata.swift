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
      envelope.roles[role] = param.name
    }

    envelopes[env] = envelope
  }
}
