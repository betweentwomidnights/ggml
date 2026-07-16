# Stable Audio 3 downstream patches

This branch is the pinned ggml dependency for
[`betweentwomidnights/sa3.cpp`](https://github.com/betweentwomidnights/sa3.cpp). It is based on
upstream ggml `v0.16.0` (`524f974b`) and carries focused changes needed by the native SA3
LoRA/DoRA training graphs:

1. CPU strided-source binary operations.
2. Autodiff backward support for `GGML_OP_CONCAT`.
3. CUDA strided-source/destination unary operations.
4. Additional allocator free-block capacity for large functional LoRA graphs.
5. CPU and CUDA F16-weight support in `OUT_PROD` backward.
6. Contiguous materialization for strided `GGML_OP_CONT` gradients.
7. Vulkan `OUT_PROD` support, including the tiled shader used by LoRA backward passes.
8. A Windows CMake fix that prevents a stale `MATH_LIBRARY-NOTFOUND` cache entry from breaking
   ggml-base generation.

The functional commits are kept separate so they can be reviewed, rebased, dropped when upstreamed,
or proposed upstream independently. Do not force-push this branch after an `sa3.cpp` commit pins it;
create a new versioned patch branch when moving to a newer upstream ggml release.
