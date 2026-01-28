import XCTest
@testable import DGen

/// Metal backend tests using shared test cases.
/// Each test runs the exact same graph as the C backend tests.
final class MetalTensorOpsTests: XCTestCase {

    // MARK: - Basic Operations

    func testSumReduceExecution() throws {
        try runMetalTest(TensorTestCases.sumReduceExecution)
    }

    func testTensorAddScalarExecution() throws {
        try runMetalTest(TensorTestCases.tensorAddScalarExecution)
    }

    func testTensorMulTensor() throws {
        try runMetalTest(TensorTestCases.tensorMulTensor)
    }

    func testBroadcastScalarTensor() throws {
        try runMetalTest(TensorTestCases.broadcastScalarTensor)
    }

    // MARK: - Tensor History

    func testTensorHistoryExecution() throws {
        try runMetalTest(TensorTestCases.tensorHistoryExecution)
    }

    // MARK: - Conv2d

    func testConv2dExecution() throws {
        try runMetalTest(TensorTestCases.conv2dExecution)
    }

    // MARK: - Conv1d

    func testConv1dExecution() throws {
        try runMetalTest(TensorTestCases.conv1dExecution)
    }

    func testConv1dIdentityKernel() throws {
        try runMetalTest(TensorTestCases.conv1dIdentityKernel)
    }

    // MARK: - Reshape

    func testReshapeExecution() throws {
        try runMetalTest(TensorTestCases.reshapeExecution)
    }

    func testReshapeToFlat() throws {
        try runMetalTest(TensorTestCases.reshapeToFlat)
    }

    func testReshapeFromFlat() throws {
        try runMetalTest(TensorTestCases.reshapeFromFlat)
    }

    func testReshapeThenSumAxisExecution() throws {
        try runMetalTest(TensorTestCases.reshapeThenSumAxisExecution)
    }

    // MARK: - SumAxis

    func testSumAxisExecution() throws {
        try runMetalTest(TensorTestCases.sumAxisExecution)
    }

    func testSumAxisAxis0() throws {
        try runMetalTest(TensorTestCases.sumAxisAxis0)
    }

    func testSumAxisToScalar() throws {
        try runMetalTest(TensorTestCases.sumAxisToScalar)
    }

    func testNestedParallelRangeDebug() throws {
        try runMetalTest(TensorTestCases.nestedParallelRangeDebug)
    }

    // MARK: - Matmul

    func testMatmulExecution() throws {
        try runMetalTest(TensorTestCases.matmulExecution)
    }

    func testMatmulWithScalarMul() throws {
        try runMetalTest(TensorTestCases.matmulWithScalarMul)
    }

    // MARK: - Transpose

    func testTransposeExecution() throws {
        try runMetalTest(TensorTestCases.transposeExecution)
    }

    // MARK: - Pad

    func testPadExecution() throws {
        try runMetalTest(TensorTestCases.padExecution)
    }

    func testPadAsymmetric() throws {
        try runMetalTest(TensorTestCases.padAsymmetric)
    }

    func testConcatViaPadAndAdd() throws {
        try runMetalTest(TensorTestCases.concatViaPadAndAdd)
    }

    func testConcat2DViaPadAndAdd() throws {
        try runMetalTest(TensorTestCases.concat2DViaPadAndAdd)
    }

    // MARK: - Shrink

    func testShrinkExecution() throws {
        try runMetalTest(TensorTestCases.shrinkExecution)
    }

    func testShrinkColumnOnly() throws {
        try runMetalTest(TensorTestCases.shrinkColumnOnly)
    }

    func testShrinkWithScalarOp() throws {
        try runMetalTest(TensorTestCases.shrinkWithScalarOp)
    }

    func testShrinkWithSumAxis() throws {
        try runMetalTest(TensorTestCases.shrinkWithSumAxis)
    }

    func testChainedShrink() throws {
        try runMetalTest(TensorTestCases.chainedShrink)
    }

    func testShrinkWithBroadcastOp() throws {
        try runMetalTest(TensorTestCases.shrinkWithBroadcastOp)
    }

    func testShrinkThenConv2d() throws {
        try runMetalTest(TensorTestCases.shrinkThenConv2d)
    }

    // MARK: - Stack

    func testStackBasic() throws {
        try runMetalTest(TensorTestCases.stackBasic)
    }

    func testStackWithShape() throws {
        try runMetalTest(TensorTestCases.stackWithShape)
    }

    // MARK: - Latch

    func testLatchWithTensorInputs() throws {
        try runMetalTest(TensorTestCases.latchWithTensorInputs)
    }

    // MARK: - Phasor with Tensor

    func testPhasorWithTensorFrequencies() throws {
        try runMetalTest(TensorTestCases.phasorWithTensorFrequencies)
    }

    // MARK: - Accum with Tensor

    func testAccumWithTensorInputs() throws {
        try runMetalTest(TensorTestCases.accumWithTensorInputs)
    }

    // MARK: - Cos Phasor Tensor

    func testCosPhasorTensor() throws {
        try runMetalTest(TensorTestCases.cosPhasorTensor)
    }

    func testCosPhasorTensorLarge() throws {
        try runMetalTest(TensorTestCases.cosPhasorTensorLarge)
    }

    // MARK: - Peek

    func testPeekOnPhasorTensor() throws {
        try runMetalTest(TensorTestCases.peekOnPhasorTensor)
    }

    // MARK: - Complex Simulation

    func testMembraneSimulationExecute() throws {
        try runMetalTest(TensorTestCases.membraneSimulationExecute)
    }
}
