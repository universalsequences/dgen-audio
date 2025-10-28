import Foundation

public enum DGenError: Error, LocalizedError {
    case insufficientInputs(operator: String, expected: Int, actual: Int)
    case missingVariableID
    case missingTensorID
    case invalidNodeReference
    case compilationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .insufficientInputs(let op, let expected, let actual):
            return "\(op) requires \(expected) inputs but got \(actual)"
        case .missingVariableID:
            return "Variable ID is missing"
        case .missingTensorID:
            return "Tensor ID is missing"
        case .invalidNodeReference:
            return "Invalid node reference"
        case .compilationFailed(let message):
            return "Compilation failed: \(message)"
        }
    }
}
