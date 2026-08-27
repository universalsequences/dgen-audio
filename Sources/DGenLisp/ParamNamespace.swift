/// Stable identity for a DGenLisp parameter.
///
/// Source declarations retain a short name. A group, when present, supplies the
/// single namespace component used by source references and host manifests.
struct ParamIdentity: Hashable {
    let shortName: String
    let group: String?

    var canonicalName: String {
        guard let group else { return shortName }
        return "\(group).\(shortName)"
    }
}

/// Declaration and reference index for parameter names.
///
/// Parameters are unique by `(group, shortName)`. Dotted references select a
/// canonical identity exactly; bare references remain backwards-compatible
/// only while their short name is unique across all groups.
struct ParamNamespace {
    private var identities: Set<ParamIdentity> = []
    private var byCanonicalName: [String: ParamIdentity] = [:]
    private var byShortName: [String: [ParamIdentity]] = [:]

    mutating func removeAll() {
        identities.removeAll()
        byCanonicalName.removeAll()
        byShortName.removeAll()
    }

    mutating func declare(shortName: String, group: String?) throws -> ParamIdentity {
        let identity = ParamIdentity(shortName: shortName, group: group)
        if identities.contains(identity) {
            throw LispError.validationError(
                "duplicate param '\(identity.canonicalName)': parameter identity must be unique by (group, name)"
            )
        }
        if let existing = byCanonicalName[identity.canonicalName] {
            throw LispError.validationError(
                "params ('\(existing.group ?? "ungrouped")', '\(existing.shortName)') and "
                + "('\(group ?? "ungrouped")', '\(shortName)') produce the same canonical id "
                + "'\(identity.canonicalName)'"
            )
        }
        identities.insert(identity)
        byCanonicalName[identity.canonicalName] = identity
        byShortName[shortName, default: []].append(identity)
        return identity
    }

    func identity(shortName: String, group: String?) -> ParamIdentity? {
        let identity = ParamIdentity(shortName: shortName, group: group)
        return byCanonicalName[identity.canonicalName]
    }

    /// Resolve a source reference. A missing bare name is not necessarily a
    /// parameter and returns nil; a missing dotted name is an invalid exact
    /// parameter reference when `requiresParameter` is true.
    func resolve(_ reference: String, requiresParameter: Bool = false) throws -> ParamIdentity? {
        if reference.contains(".") {
            if let identity = byCanonicalName[reference] {
                return identity
            }
            if requiresParameter {
                throw LispError.validationError(
                    "parameter reference '\(reference)' does not match a declared group.name identity"
                )
            }
            return nil
        }

        let candidates = byShortName[reference] ?? []
        if candidates.count == 1 {
            return candidates[0]
        }
        if candidates.count > 1 {
            let names = candidates.map(\.canonicalName).sorted().joined(separator: ", ")
            throw LispError.validationError(
                "ambiguous parameter reference '\(reference)'; use one of: \(names)"
            )
        }
        if requiresParameter {
            throw LispError.validationError("'\(reference)' does not reference a parameter")
        }
        return nil
    }
}
