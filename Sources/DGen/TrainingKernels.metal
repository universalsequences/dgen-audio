#include <metal_stdlib>
using namespace metal;

// MARK: - Gradient Reduction

/// Reduce gradients across all frames for each gradient ID
/// Input: gradients[frameCount * gradId + frameIndex]
/// Output: reducedGrads[gradId] = mean(gradients for this gradId)
kernel void reduceGradients(
    device const float* gradients [[buffer(0)]],
    device float* reducedGrads [[buffer(1)]],
    constant int& frameCount [[buffer(2)]],
    constant int& numGradIds [[buffer(3)]],
    uint gradId [[thread_position_in_grid]]
) {
    if (gradId >= numGradIds) return;

    // Sum across all frames for this gradId
    float sum = 0.0;
    int baseIdx = frameCount * gradId;
    for (int i = 0; i < frameCount; i++) {
        float g = gradients[baseIdx + i];
        // Be robust to any accidental NaNs/Infs in per-frame grads
        if (!isfinite(g)) { g = 0.0; }
        sum += g;
    }

    // Store mean gradient
    reducedGrads[gradId] = sum / float(frameCount);
}

// MARK: - Parameter Updates

/// Update parameters using SGD optimizer
/// memory[physicalCells[i]] -= learningRate * reducedGrads[gradIds[i]]
kernel void updateParametersSGD(
    device float* memory [[buffer(0)]],
    device const float* reducedGrads [[buffer(1)]],
    constant int* gradIds [[buffer(2)]],
    constant int* physicalCells [[buffer(3)]],
    constant float& learningRate [[buffer(4)]],
    constant int& paramCount [[buffer(5)]],
    uint paramIdx [[thread_position_in_grid]]
) {
    if (paramIdx >= paramCount) return;

    int gradId = gradIds[paramIdx];
    int physicalCell = physicalCells[paramIdx];
    float grad = reducedGrads[gradId];

    // SGD update: param -= lr * grad
    memory[physicalCell] -= learningRate * grad;
}

/// Update parameters using Adam optimizer
/// Implements bias-corrected Adam with momentum
kernel void updateParametersAdam(
    device float* memory [[buffer(0)]],
    device const float* reducedGrads [[buffer(1)]],
    device float* m [[buffer(2)]],                // First moment
    device float* v [[buffer(3)]],                // Second moment
    constant int* gradIds [[buffer(4)]],
    constant int* physicalCells [[buffer(5)]],
    constant float& learningRate [[buffer(6)]],
    constant float& beta1 [[buffer(7)]],
    constant float& beta2 [[buffer(8)]],
    constant float& epsilon [[buffer(9)]],
    constant int& timestep [[buffer(10)]],
    constant int& paramCount [[buffer(11)]],
    uint paramIdx [[thread_position_in_grid]]
) {
    if (paramIdx >= paramCount) return;

    int gradId = gradIds[paramIdx];
    int physicalCell = physicalCells[paramIdx];
    float grad = reducedGrads[gradId];

    // Update biased first moment estimate
    m[paramIdx] = beta1 * m[paramIdx] + (1.0 - beta1) * grad;

    // Update biased second raw moment estimate
    v[paramIdx] = beta2 * v[paramIdx] + (1.0 - beta2) * grad * grad;

    // Compute bias-corrected first moment estimate
    float m_hat = m[paramIdx] / (1.0 - pow(beta1, float(timestep)));

    // Compute bias-corrected second raw moment estimate
    float v_hat = v[paramIdx] / (1.0 - pow(beta2, float(timestep)));

    // Update parameter
    memory[physicalCell] -= learningRate * m_hat / (sqrt(v_hat) + epsilon);
}
