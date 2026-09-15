# rotorch (formerly gato) vs this repo's CGENN -- what was taken, what was not

Compared at rotorch `0970e0e` (tBuLi/rotorch, kingdon `feature/torch_backend`) against
`experiments/baselines/cgenn`. Full comparison in the session notes; this records the
decisions that landed here and the ones deliberately not taken.

## Landed on `gato-lessons` (tests and tooling only -- NO runtime code path changes)

- `test_cgenn_compile.py::test_fullgraph_compiles`: `fullgraph=True` with `dynamic=True`
  kept. rotorch traces its whole model as one graph; this pins that the net here does too,
  per gp_impl. Gated like the other compile gates (CGENN_COMPILE_GATES=1).
- `test_cgenn_grade_structure.py`: exact-zero asserts on blades no product can reach after
  the embedding and the first message product (the first node update already reaches every
  grade through bivector x bivector) -- the dense-tensor version
  of rotorch's output-type asserts. Catches a sign-table, hoist-slice or kernel-slot bug
  with exact arithmetic, on CPU, for every gp_impl.
- `test_readout_batch_dependence.py`: pins that a jet's score is independent of its batch
  mates at fixed padded width (eval reproducibility) and that it DOES depend on the padded
  width (the frozen official-repo readout quirk), so either property can only change on
  purpose.
- `utils/cost_fit.py`: rotorch's fixed-plus-marginal step-cost fit, for the race scripts.
  Reproduces the numbers in rotorch's benchmark doc from its own table.

Numerics: unchanged, bit for bit -- nothing on a runtime path moved. Speed: unchanged.

## Not taken, with the estimate

- **Grade-aware first layer.** rotorch allocates only reachable grade paths per layer. Here
  the shipped `gp_impl=flash` kernel is generated for all 35 Cl(1,3) paths, so the saving
  needs a second generated kernel for vector-only operands (2 paths: (1,1,0), (1,1,2)) and
  its own parity gates. Only layer 1's `phi_x` qualifies (from layer 2 every grade is
  populated, as rotorch's own lorentz benchmark notes); the sparse path would gain the same
  way by slicing its tables. Bound on the win: one of eight products per forward at ~17x
  fewer MACs, i.e. under ~10% of product time and a few percent of the step. Worth a kernel
  variant later, not this change.
- **Invariants only for present grades.** Changes `phi_h`'s input width, hence parameter
  shapes and checkpoints: a modeling change.
- **Mask-aware readout / BatchNorm.** Frozen as official-repo parity (docs/cgenn-compile.md).
  `test_readout_batch_dependence.py` now pins the dependence so a future change is visible.
