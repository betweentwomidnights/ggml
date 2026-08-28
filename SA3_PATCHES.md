# Shared SA3 / ACE-Step downstream patches

This 0.17-based candidate is intended as the shared dependency of
[`betweentwomidnights/sa3.cpp`](https://github.com/betweentwomidnights/sa3.cpp) and
[`betweentwomidnights/acestep.cpp`](https://github.com/betweentwomidnights/acestep.cpp).
It retains the ACE training branch's upstream history and the SA3 training ports,
including Twilwa's commits and the original authors of the ported changes.
The previous 0.16 training history is also retained as a merge parent.

1. CPU strided-source binary operations.
2. Autodiff backward support for `GGML_OP_CONCAT`.
3. CUDA strided-source/destination unary operations.
4. Additional allocator free-block capacity for large functional LoRA graphs.
5. CPU and CUDA F16-weight support in `OUT_PROD` backward.
6. Contiguous materialization for strided `GGML_OP_CONT` gradients.
7. Vulkan `OUT_PROD` support, including the tiled shader used by LoRA backward passes.
8. A Windows CMake fix that prevents a stale `MATH_LIBRARY-NOTFOUND` cache entry from breaking
   ggml-base generation.
9. Metal native backward operations and simdgroup-tiled `OUT_PROD`.
10. Metal non-inplace `SET` wide-row copy fix.
11. Quantized frozen weights in CUDA, Vulkan, and Metal `OUT_PROD` (Q4_K, Q5_K,
    Q6_K, Q8_0), including Vulkan's paired tile loads.
12. GGUF tensor-dimension accessor compatibility.
13. Metal `OUT_PROD` simdgroup-matrix capability guard for unsupported older devices.

## Candidate provenance and validation

The computational source is exactly ACE's `d1bead82` plus the Metal capability
guard ported from SA3's `3d9d897a`. The history reconciliation removes only stale
duplicate CUDA get-rows helpers and GGUF accessors reintroduced by merging the
older branch; it does not alter the validated candidate's code. This document is
the only tree difference from that validated source.

Windows RTX 5070 Laptop validation (2026-08-27):

- SA3 main `378c269`: complete CUDA and Vulkan builds, 18 CTests per backend and
  revision, inference on F16/Q4 small-music and medium, functional adapters/blends,
  native preprocessing/training/resume, and C-ABI training.
- Both backends: 15 waveform pairs and all compared adapter/optimizer tensors
  exactly match the 0.16 baseline. This is within-backend equality, not a claim
  that CUDA and Vulkan generate the same samples.
- 52 model-backed SA3 runs per backend; no meaningful performance improvement.
- ACE: complete Vulkan build, nine CTests, 60 functional-adapter GPU cases, five
  DoRA forward/backward weight types, and XL-base Q4 synthesis with/without a
  known-good functional DoRA adapter. Earlier CUDA application checks also passed.

Metal runtime/training on the M4 is still a required gate. The Metal guard has not
been compiled by Windows validation. AMD/Intel Vulkan and long-run training quality
are not established by these tests. Keep the application PRs unmerged until their
Metal checks pass against the same exact submodule commit.

The functional commits remain reviewable with their original attribution. Do not
force-push a branch after either application pins it; use a new commit or versioned
branch for subsequent changes.
