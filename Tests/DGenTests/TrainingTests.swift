import XCTest

@testable import DGen

final class TrainingTests: XCTestCase {

    /// Test that a simple scalar parameter can be learned via gradient descent
    /// We'll learn a constant value: target = 0.5
    func testLearnScalarGain() throws {
        print("\n🧪 Test: Learn scalar parameter to match constant")

        // MARK: - Build Graph

        let g = Graph()

        // Learnable parameter (start at wrong value: 0.1, target: 0.5)
        let learnableValue = Parameter(graph: g, value: 0.1, name: "value")

        // Output is just the learnable value
        let output = learnableValue.node()

        // Target: constant 0.5
        let target = g.n(.constant(0.5))

        // Loss function (MSE)
        let loss = g.n(.mse, output, target)

        // Route output
        _ = g.n(.output(0), loss)

        print("   Initial value: \(learnableValue.value)")

        // MARK: - Compile

        let frameCount = 128
        let result = try CompilationPipeline.compile(
            graph: g,
            backend: .metal,
            options: .init(frameCount: frameCount, debug: false, backwards: true)
        )

        let runtime = try MetalCompiledKernel(
            kernels: result.kernels,
            cellAllocations: result.cellAllocations,
            context: result.context
        )

        print("   ✅ Compiled with backwards pass enabled")
        print("\n=== METAL KERNELS ===")
        for (i, kernel) in result.kernels.enumerated() {
            print("\n--- Kernel \(i): \(kernel.name) ---")
            print(kernel.source)
        }
        print("=== END KERNELS ===\n")

        // MARK: - Training Setup

        let ctx = TrainingContext(
            parameters: [learnableValue],
            optimizer: SGD(lr: 0.01),
            lossNode: loss
        )

        ctx.initializeMemory(
            runtime: runtime,
            cellAllocations: result.cellAllocations,
            context: result.context,
            frameCount: frameCount
        )

        // MARK: - Training Loop

        let numEpochs = 200
        var finalLoss: Float = 0.0

        for epoch in 0..<numEpochs {
            let loss = ctx.runStepGPU()
            finalLoss = loss

            // DEBUG: Check memory buffer to confirm parameter was updated
            if epoch % 20 == 0 {
                if let memBuffer = runtime.getBuffer(name: "memory") {
                    let memPtr = memBuffer.contents().assumingMemoryBound(to: Float.self)
                    let physicalCell =
                        result.cellAllocations.cellMappings[learnableValue.cellId]
                        ?? learnableValue.cellId
                    let memValue = memPtr[physicalCell]
                    print(
                        "   Epoch \(epoch): param.value = \(String(format: "%.4f", learnableValue.value)), memory[\(physicalCell)] = \(String(format: "%.4f", memValue)), loss = \(String(format: "%.6f", loss))"
                    )
                } else {
                    print(
                        "   Epoch \(epoch): value = \(String(format: "%.4f", learnableValue.value)), loss = \(String(format: "%.6f", loss))"
                    )
                }
            }
        }

        // MARK: - Verification

        print("   Final value: \(String(format: "%.4f", learnableValue.value))")
        print("   Target value: 0.5000")
        print("   Final loss: \(String(format: "%.6f", finalLoss))")

        // Assert the parameter converged close to 0.5
        XCTAssertEqual(learnableValue.value, 0.5, accuracy: 0.05, "Value should converge to ~0.5")

        // Assert loss is small
        XCTAssertLessThan(finalLoss, 0.001, "Final loss should be < 0.001")

        print("   ✅ Test passed: learned value = \(String(format: "%.4f", learnableValue.value))")
    }

    /// Test learning multiple parameters simultaneously
    func testLearnMultipleParameters() throws {
        print("\n🧪 Test: Learn multiple parameters (sum to target)")

        // MARK: - Build Graph

        let g = Graph()

        // Two learnable parameters (start at DIFFERENT values to break symmetry)
        let param1 = Parameter(graph: g, value: 0.2, name: "param1")
        let param2 = Parameter(graph: g, value: 0.05, name: "param2")

        // Output = param1 + param2
        let output = g.n(.add, param1.node(), param2.node())

        // Target: constant 0.8 (so param1 + param2 should = 0.8)
        // There are infinite solutions, but SGD should find one
        let target = g.n(.constant(0.8))

        let loss = g.n(.mse, output, target)
        _ = g.n(.output(0), loss)

        let initialSum = param1.value + param2.value
        print(
            "   Initial: param1=\(param1.value), param2=\(param2.value), sum=\(initialSum) (target: 0.8)"
        )

        // MARK: - Compile & Train

        let frameCount = 128
        let result = try CompilationPipeline.compile(
            graph: g,
            backend: .metal,
            options: .init(frameCount: frameCount, backwards: true)
        )

        let runtime = try MetalCompiledKernel(
            kernels: result.kernels,
            cellAllocations: result.cellAllocations,
            context: result.context
        )

        let ctx = TrainingContext(
            parameters: [param1, param2],
            optimizer: SGD(lr: 0.1),  // Higher LR and simpler optimizer
            lossNode: loss
        )

        ctx.initializeMemory(
            runtime: runtime,
            cellAllocations: result.cellAllocations,
            context: result.context,
            frameCount: frameCount
        )

        for epoch in 0..<400 {
            let loss = ctx.runStepGPU()

            if epoch % 100 == 0 || epoch < 5 {
                let sum = param1.value + param2.value
                let expectedGrad = 2.0 * (sum - 0.25 - 0.8)  // Rough estimate
                print(
                    "   Epoch \(epoch): BEFORE_UPDATE sum=\(String(format: "%.4f", sum - (param1.value-0.2) - (param2.value-0.05))), loss=\(String(format: "%.6f", loss)) | AFTER_UPDATE sum=\(String(format: "%.4f", sum))"
                )
            }
        }

        // MARK: - Verification

        let finalSum = param1.value + param2.value
        print(
            "   Final: param1=\(String(format: "%.3f", param1.value)), param2=\(String(format: "%.3f", param2.value)), sum=\(String(format: "%.3f", finalSum))"
        )
        print("   Target sum: 0.800")

        // The sum should be close to 0.8
        XCTAssertEqual(finalSum, 0.8, accuracy: 0.1, "Sum of params should converge to ~0.8")

        print("   ✅ Test passed: learned sum = \(String(format: "%.3f", finalSum))")
    }

}
