import Foundation

public enum DGenError: Error, LocalizedError {
    case insufficientInputs(operator: String, expected: Int, actual: Int)
    case missingVariableID
    case missingTensorID
    case invalidNodeReference
    case compilationFailed(String)
    case shapeMismatch(op: String, shape1: Shape, shape2: Shape)

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
        case .shapeMismatch(let op, let shape1, let shape2):
            return "Shape mismatch in \(op): \(shape1) vs \(shape2)"
        }
    }
}
