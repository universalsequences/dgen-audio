# DGEN

A tinygrad-inspired library, written in Swift, designed to write efficient audio DSP applications, with multiple targets.

The compiler groups operations that can be done in parallel, and carefully detects codepaths that require sequential execution, such as recursive state dependencies.

## Targets
1. C (SIMD)
2. Metal

## Machine Learning
The goal for this library is for all operations to contain backwards definitions, allowing back propagation to be done.


