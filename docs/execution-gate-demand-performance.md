# Execution-gate demand propagation

Investigation: dgen-8pj, 2026-09-22, Apple M1 Max.

An isolated Digi FM source with complementary sine/table `block-gate` branches
inside its FM feedback graph failed to complete compilation in 182 seconds
with the published macOS v0.1.25 compiler. A symbolized debug-compiler sample
located the stall in `ExecutionGatePass.prepare`, with most samples beneath
`ExecutionDemand.include` sorting predicate sets.

The analysis represents demand as a disjunction of conjunctions. An empty
conjunction means unconditional execution. A broader conjunction subsumes a
narrower one containing all its predicates. The former depth-first worklist
could propagate many narrow paths through shared producers before visiting a
pending broad demand, then discard those combinations after doing the work.
Sorting the entire growing demand on each insertion amplified the cost.

The worklist now processes conjunctions in nondecreasing predicate count.
Every propagation step preserves its term or adds one predicate, so an
unprocessed shorter term cannot be generated from a longer term. This ensures
all broader demands have propagated before narrower terms are considered.
Canonical ordering, required for execution-region equality, is computed once
after convergence; each accepted set is sorted once for that comparison.

No terms are capped or dropped, and no producer is made unconditional as a
fallback. The change preserves the least fixed point and existing gate/history
semantics. It prevents transient supersets from multiplying; a graph whose
final minimal DNF is itself exponential can still be expensive.

## Validation

`swift test --filter 'ExecutionGateDemandTests|ExecutionGateTests'` passed all
16 tests. The two new tests cover:

- Twenty-two nested two-way gates sharing history, with pending unconditional
  uses of the same intermediate values. Root visitation order is deliberately
  chosen to expose the former depth-first behavior. The test takes about 3 ms
  and uses a generous five-second regression ceiling.
- Nested AND plus shared OR demand, checked for all eight assignments of three
  predicates, including history writers, dependency restoration and repeated
  analysis determinism.

Use the local hermetic stage when running native C execution tests:

```sh
DGEN_TOOLCHAIN_STAGE_ROOT=/path/to/eseq/crates/sequencer/tools/dgen-toolchain \
DGEN_RUNTIME_INCLUDE=/path/to/eseq/crates/sequencer/tools/dgen-toolchain/include \
DGEN_BINARY_AUDIT_TOOL="$PWD/scripts/audit-dgen-dylib.sh" \
swift test --filter 'ExecutionGateDemandTests|ExecutionGateTests'
```

The unchanged formerly stalled Digi FM source compiles in 3.270 seconds with
the locally built release compiler. Three other complete Digi FM sources
(original additive, table-based, and eight specialized switchable cores)
generate byte-identical C with the old and new compiler. Twenty-four static
renders and one irregular modulated render per source give 75 sample-identical
audio comparisons. Four render cases for the formerly stalled source have
finite audio/state and nonzero signal.

Detailed scripts, source snapshots, samples and results are retained in the
eseq checkout under `.local/benchmarks/digi-fm-2026-09-22/`, especially
`compiler-gate-validation.py`, `compiler-gate-validation/results.json`,
`compiler-gate-tests.log` and `compiler-blowup-debug.sample.txt`.

This change targets compile-time demand analysis. It does not change Digi FM's
waveform evaluation or reduce its audio-thread CPU cost by itself.
