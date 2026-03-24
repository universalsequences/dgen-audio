// LispParser - Tokenizer + AST parser for DGenLisp
//
// Ported from EvalLispTool.swift (patch-editor project).
// Pure string/AST operations with no DGen dependencies.

import Foundation

// MARK: - AST

enum ASTNode: Equatable {
    case atom(String)
    case list([ASTNode])
}

// MARK: - Errors

enum LispError: Error {
    case parseError(String)
    case unknownOperator(String)
    case unknownSymbol(String)
    case invalidArgument(String)
    case historyNotFound(String)
    case typeError(String)
    case validationError(String)

    var message: String {
        switch self {
        case .parseError(let msg): return "Parse error: \(msg)"
        case .unknownOperator(let name): return "Unknown operator: \(name)"
        case .unknownSymbol(let name): return "Unknown symbol: \(name)"
        case .invalidArgument(let msg): return "Invalid argument: \(msg)"
        case .historyNotFound(let name):
            return "History not found: \(name). Use (make-history \(name)) first."
        case .typeError(let msg): return "Type error: \(msg)"
        case .validationError(let msg): return "Validation error: \(msg)"
        }
    }
}

// MARK: - Parser

/// Strip comments from an expression (everything after ; or # outside of strings)
func stripComments(_ expr: String) -> String {
    let lines = expr.components(separatedBy: .newlines)
    var resultLines: [String] = []

    for line in lines {
        var lineResult = ""
        var inString = false

        for char in line {
            if char == "\"" {
                inString = !inString
                lineResult.append(char)
            } else if !inString && (char == ";" || char == "#") {
                break
            } else {
                lineResult.append(char)
            }
        }

        let trimmedLine = lineResult.trimmingCharacters(in: .whitespaces)
        if !trimmedLine.isEmpty {
            resultLines.append(trimmedLine)
        }
    }

    return resultLines.joined(separator: " ")
}

/// Tokenize a lisp expression
func tokenize(_ expr: String) -> [String] {
    var tokens: [String] = []
    var current = ""
    var inString = false
    var inBracket = false
    var bracketDepth = 0

    for char in expr {
        if char == "\"" && !inBracket {
            inString = !inString
            current.append(char)
        } else if inString {
            current.append(char)
        } else if char == "[" && !inString {
            if !inBracket {
                if !current.isEmpty {
                    tokens.append(current)
                    current = ""
                }
                inBracket = true
                bracketDepth = 1
            } else {
                bracketDepth += 1
            }
            current.append(char)
        } else if char == "]" && inBracket {
            current.append(char)
            bracketDepth -= 1
            if bracketDepth == 0 {
                inBracket = false
                tokens.append(current)
                current = ""
            }
        } else if inBracket {
            current.append(char)
        } else if char == "(" || char == ")" {
            if !current.isEmpty {
                tokens.append(current)
                current = ""
            }
            tokens.append(String(char))
        } else if char.isWhitespace {
            if !current.isEmpty {
                tokens.append(current)
                current = ""
            }
        } else {
            current.append(char)
        }
    }

    if !current.isEmpty {
        tokens.append(current)
    }

    return tokens
}

/// Parse tokens into an AST node
func parse(_ tokens: inout ArraySlice<String>) throws -> ASTNode {
    guard let token = tokens.popFirst() else {
        throw LispError.parseError("Unexpected end of expression")
    }

    if token == "(" {
        var list: [ASTNode] = []
        while tokens.first != ")" {
            if tokens.isEmpty {
                throw LispError.parseError("Missing closing parenthesis")
            }
            list.append(try parse(&tokens))
        }
        tokens.removeFirst()  // consume ")"
        return .list(list)
    } else if token == ")" {
        throw LispError.parseError("Unexpected closing parenthesis")
    } else {
        return .atom(token)
    }
}

// MARK: - Binary Operator Nesting

private let binaryOperators: Set<String> = ["+", "-", "*", "/", "%", "min", "max", "and", "or"]

/// Transform n-ary binary operators into nested binary form (left-associative)
/// e.g., (+ a b c d) becomes (+ (+ (+ a b) c) d)
func nestifyBinaryOperators(_ node: ASTNode) -> ASTNode {
    switch node {
    case .atom:
        return node

    case .list(let elements):
        guard !elements.isEmpty else { return node }

        let transformedElements = elements.map { nestifyBinaryOperators($0) }

        guard case .atom(let opName) = transformedElements[0],
            binaryOperators.contains(opName.lowercased())
        else {
            return .list(transformedElements)
        }

        let args = Array(transformedElements.dropFirst())

        guard args.count > 2 else {
            return .list(transformedElements)
        }

        var result: ASTNode = .list([.atom(opName), args[0], args[1]])
        for i in 2..<args.count {
            result = .list([.atom(opName), result, args[i]])
        }

        return result
    }
}

// MARK: - Convenience

/// Parse a complete source string into a list of top-level AST nodes
func parseSource(_ source: String) throws -> [ASTNode] {
    let cleaned = stripComments(source)
    let tokens = tokenize(cleaned)
    var slice = ArraySlice(tokens)

    var nodes: [ASTNode] = []
    while !slice.isEmpty {
        let ast = try parse(&slice)
        nodes.append(nestifyBinaryOperators(ast))
    }
    return nodes
}
