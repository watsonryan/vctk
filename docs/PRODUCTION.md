# VCTK Production Operations Guide

## Threading Model

- `learnVDP`, `learnBGMM`, `learnDGMM`, and `learnBEMM` are safe to run concurrently
  on separate output objects (`qZ`, `weights`, `clusters`).
- OpenMP thread count is controlled per parallel region via `VbemOptions.n_threads`.
- `deterministic=true` forces a deterministic single-thread execution path.

## Exception Contract

Public C++ APIs throw:

- `std::invalid_argument` for invalid shapes, ranges, or non-finite inputs.
- `vctk::RuntimeError` (derived from `std::runtime_error`) for runtime fitting failures.

`vctk::RuntimeError` exposes:

- `code()` -> `vctk::ErrorCode`
- `context()` -> `vctk::ErrorContext` (`algorithm`, `phase`, iteration, clusters, free energy)

Strong exception guarantee:

- output references (`qZ`, `weights`, `clusters`) are unchanged if an exception is thrown.

## Heuristic Tuning Controls

Key split/prune controls in `VbemOptions`:

- `min_split_cluster_obs`: minimum cluster size before split is considered.
- `min_split_partition_obs`: minimum observations per proposed split side.
- `min_split_improvement`: required relative free-energy improvement for accepting a split.
- `prune_zero_cutoff`: pruning threshold for empty/near-empty clusters.

Recommended starting values:

- Small datasets: `min_split_cluster_obs=4`, `min_split_partition_obs=2`, `min_split_improvement=1e-5`
- Large/noisy datasets: increase `min_split_cluster_obs` and `min_split_improvement`.

## Observability

- `progress_callback`: per-outer-iteration telemetry (`LearnProgress`).
- `error_callback`: structured failure telemetry (`ErrorContext`), callback exceptions are swallowed.

## C API Boundary

Use `include/vctk_c.h`:

- `vctk_learn_vdp_labels(...)`
- returns status code (`VCTK_STATUS_*`) and optional error message text.
- never throws C++ exceptions across the boundary.
