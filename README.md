# vctk
variational cluster toolkit

## Quick Start (VDP)

```cpp
#include <vctk.hpp>

#include <Eigen/Dense>
#include <vector>

#include "distributions.hpp"

Eigen::MatrixXd residuals; // NxD data
Eigen::MatrixXd qZ;
distributions::StickBreak weights;
std::vector<distributions::GaussWish> clusters;

const double F = vctk::learnVDP(residuals, qZ, weights, clusters);
```

`VbemOptions` includes production controls:
- `deterministic=true`: reproducible execution mode
- `n_threads`: explicit OpenMP thread count (0 = runtime default)
- `min_split_cluster_obs`, `min_split_partition_obs`: split sensitivity controls
- `min_split_improvement`: minimum relative free-energy gain to accept a split
- `prune_zero_cutoff`: pruning sensitivity control
- `progress_callback`: structured per-outer-iteration progress events
- `error_callback`: structured error context telemetry on runtime failures
- `random_seed`: reserved for stochastic extensions

Public exception contract (`learnVDP`, `learnBGMM`, `learnDGMM`, `learnBEMM`):
- `std::invalid_argument`: invalid shapes, ranges, or non-finite inputs
- `vctk::RuntimeError`: numerical/optimization failures (`ErrorCode` + `ErrorContext`)
- Strong exception guarantee: output references are unchanged on exception

C API boundary:
- Header: `include/vctk_c.h`
- Function: `vctk_learn_vdp_labels(...)`
- Returns status code + optional error message buffer instead of C++ exceptions

Operations guide:
- `docs/PRODUCTION.md`

## Merge API (integrated)

```cpp
#include <merge.hpp>

std::vector<vctk::merge::MixtureComponent> prior;
auto merged = vctk::merge::mergeMixtureModel(
    residuals, qZ, prior, clusters, weights, 0.05, 32);
```

## Example: Cluster + Merge + Plot

Build and run the VDP + merge example:

```sh
cmake --preset macos-debug -DVCTK_BUILD_EXAMPLES=ON
cmake --build --preset build-macos-debug --target vctk_vdp_example
./build/macos-debug/vctk_vdp_example
```

Run explicit merge proof (2 clusters, then 3 with 2 overlaps):

```sh
cmake --preset macos-debug --fresh -DVCTK_BUILD_EXAMPLES=ON
cmake --build --preset build-macos-debug --target vctk_merge_proof_example
./build/macos-debug/vctk_merge_proof_example
```

Generate a publication-style figure (Python 3):

```sh
python3 examples/plot_vdp.py \
  --points build/examples/vdp_points.csv \
  --centers build/examples/vdp_merged_centers.csv \
  --out build/examples/vdp_plot.pdf
```

## Build with CMake presets (macOS native)

Configure + build debug:

```sh
cmake --preset macos-debug
cmake --build --preset build-macos-debug
```

Configure + build release:

```sh
cmake --preset macos-release
cmake --build --preset build-macos-release
```

Run tests:

```sh
ctest --preset test-macos-debug
ctest --preset test-macos-release
```

Algorithm tests now include:
- deterministic regression checks (repeatability on fixed fixtures)
- parallel independent-run stress checks (`std::thread`)
- invalid numeric input rejection checks (NaN/Inf)

Sanitizer build (ASan + UBSan):

```sh
cmake --preset macos-sanitize
cmake --build --preset build-macos-sanitize
ctest --preset test-macos-sanitize
```

## Build with CMake presets (Linux native)

Configure + build debug:

```sh
cmake --preset linux-debug
cmake --build --preset build-linux-debug
```

Configure + build release:

```sh
cmake --preset linux-release
cmake --build --preset build-linux-release
```

Run tests:

```sh
ctest --preset test-linux-debug
ctest --preset test-linux-release
```

Sanitizer build (ASan + UBSan):

```sh
cmake --preset linux-sanitize
cmake --build --preset build-linux-sanitize
ctest --preset test-linux-sanitize
```

## Benchmarking (VDP)

Enable benchmark targets and build:

```sh
cmake --preset macos-release --fresh -DVCTK_BUILD_BENCHMARKS=ON
cmake --build --preset build-macos-release -j
```

Run benchmark sweeps over thread counts:

```sh
./build/macos-release/vctk_bench_vdp \
  --n 20000 \
  --d 10 \
  --k 10 \
  --reps 5 \
  --threads 1,2,4,8 \
  --csv build/bench/vdp_bench.csv
```

CSV columns:
- `n,d,k_true`: synthetic dataset shape and generating clusters
- `threads`: threads passed to `vctk::VbemOptions.n_threads`
- `run_idx`: repetition index
- `clusters_found`: inferred cluster count
- `free_energy`: converged objective
- `elapsed_ms`: total `learnVDP` runtime in ms
