#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

// Low mantissa bits matter: reduced-precision source conversion must fail this
// FP64 dot-product oracle even when the kernel accumulates its products in F32.
static bool test_shape(ggml_backend_t backend, int n, bool explicit_acc) {
    const int k = 512;
    const int m = 64;
    std::vector<float> a(k*m), b(k*n), actual(m*n);
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = 0.3f*std::sin(float(i)*0.137f) + 0.00013f;
    }
    for (size_t i = 0; i < b.size(); ++i) {
        b[i] = 0.2f*std::cos(float(i)*0.171f) - 0.00017f;
    }

    ggml_context * ctx = ggml_init({4*1024*1024, nullptr, true});
    if (!ctx) {
        return false;
    }
    ggml_tensor * ta = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
    ggml_tensor * tb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    ggml_tensor * out = ggml_mul_mat(ctx, ta, tb);
    bool passed = ggml_prec_set_src(out, GGML_PREC_F32, 1);
    if (explicit_acc) {
        passed = ggml_prec_set_acc(out, GGML_PREC_F32) && passed;
    }
    if (!ggml_backend_supports_op(backend, out)) {
        std::fprintf(stderr, "%s does not support the requested F32 matmul\n", ggml_backend_name(backend));
        ggml_free(ctx);
        return false;
    }
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(ta, a.data(), 0, a.size()*sizeof(float));
    ggml_backend_tensor_set(tb, b.data(), 0, b.size()*sizeof(float));
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size()*sizeof(float));

    double err = 0.0, ref = 0.0, maximum = 0.0;
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            double expected = 0.0;
            for (int i = 0; i < k; ++i) {
                expected += double(a[row*k + i])*b[col*k + i];
            }
            const double delta = double(actual[col*m + row]) - expected;
            passed = std::isfinite(actual[col*m + row]) && passed;
            err += delta*delta;
            ref += expected*expected;
            maximum = std::fmax(maximum, std::fabs(delta));
        }
    }
    const double relative = std::sqrt(err/ref);
    passed = passed && relative < 1e-5 && maximum < 1e-5;
    std::printf("%s F32 src1, %s acc, K=%d M=%d N=%d: max_abs=%.9g relative_l2=%.9g %s\n",
        ggml_backend_name(backend), explicit_acc ? "F32" : "default", k, m, n,
        maximum, relative, passed ? "PASS" : "FAIL");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return passed;
}

// Attention-shaped operands: both are permuted views [k, heads, rows] -> [k, rows, heads], so dims 0/1
// are not contiguous and a backend must stage them without dropping to F16.
static bool test_permuted(ggml_backend_t backend, int n) {
    const int k = 128, m = 37, heads = 4;
    std::vector<float> a(k*heads*m), b(k*heads*n), actual(m*n*heads);
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = 0.3f*std::sin(float(i)*0.113f) + 0.00011f;
    }
    for (size_t i = 0; i < b.size(); ++i) {
        b[i] = 0.2f*std::cos(float(i)*0.157f) - 0.00019f;
    }
    ggml_context * ctx = ggml_init({4*1024*1024, nullptr, true});
    if (!ctx) {
        return false;
    }
    ggml_tensor * ta = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, heads, m);
    ggml_tensor * tb = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, heads, n);
    ggml_tensor * out = ggml_mul_mat(ctx, ggml_permute(ctx, ta, 0, 2, 1, 3), ggml_permute(ctx, tb, 0, 2, 1, 3));
    bool passed = ggml_prec_set_src(out, GGML_PREC_F32, 1) && ggml_prec_set_acc(out, GGML_PREC_F32);
    if (!ggml_backend_supports_op(backend, out)) {
        std::fprintf(stderr, "%s does not support the requested permuted F32 matmul\n", ggml_backend_name(backend));
        ggml_free(ctx);
        return false;
    }
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(ta, a.data(), 0, a.size()*sizeof(float));
    ggml_backend_tensor_set(tb, b.data(), 0, b.size()*sizeof(float));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size()*sizeof(float));

    double err = 0.0, ref = 0.0, maximum = 0.0;
    for (int h = 0; h < heads; ++h) {
        for (int col = 0; col < n; ++col) {
            for (int row = 0; row < m; ++row) {
                double expected = 0.0;
                for (int i = 0; i < k; ++i) {
                    expected += double(a[(size_t(row)*heads + h)*k + i])*b[(size_t(col)*heads + h)*k + i];
                }
                const float got = actual[(size_t(h)*n + col)*m + row];
                const double delta = double(got) - expected;
                passed = std::isfinite(got) && passed;
                err += delta*delta;
                ref += expected*expected;
                maximum = std::fmax(maximum, std::fabs(delta));
            }
        }
    }
    const double relative = std::sqrt(err/ref);
    passed = passed && relative < 1e-5 && maximum < 1e-5;
    std::printf("%s permuted F32 src1, F32 acc, K=%d M=%d N=%d heads=%d: max_abs=%.9g relative_l2=%.9g %s\n",
        ggml_backend_name(backend), k, m, n, heads, maximum, relative, passed ? "PASS" : "FAIL");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return passed;
}

// Vulkan honors an F32 src1 only with F32 accumulation (its F32 x F32 path); with default accumulation it
// must refuse the op, so a caller without a scheduler fallback fails loudly instead of losing precision.
static bool vulkan_refuses_default_acc(ggml_backend_t backend) {
    ggml_context * ctx = ggml_init({1024*1024, nullptr, true});
    ggml_tensor * out = ggml_mul_mat(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 64),
                                     ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 8));
    ggml_prec_set_src(out, GGML_PREC_F32, 1);
    const bool refused = !ggml_backend_supports_op(backend, out);
    std::printf("%s F32 src1, default acc: %s\n", ggml_backend_name(backend), refused ? "refused PASS" : "accepted FAIL");
    ggml_free(ctx);
    return refused;
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    bool passed = true, tested = false;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        // This regression targets CUDA dispatch and its CPU oracle, plus Vulkan's F32 x F32 path. Default
        // accumulation is not a portable F32 guarantee for other backends.
        const std::string name = ggml_backend_dev_name(dev);
        const bool vulkan = name.rfind("Vulkan", 0) == 0;
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU && name.rfind("CUDA", 0) != 0 && !vulkan) {
            continue;
        }
        if (argc > 1 && std::string(ggml_backend_dev_name(dev)) != argv[1]) {
            continue;
        }
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) {
            continue;
        }
        tested = true;
        if (vulkan) {
            passed = vulkan_refuses_default_acc(backend) && passed;
        }
        // MMVF, MMF interior/boundaries, and the ordinary GEMM path (Vulkan: mat-vec and mat-mat).
        for (int n : {1, 5, 8, 16, 17, 25, 64}) {
            for (bool explicit_acc : {false, true}) {
                if (vulkan && !explicit_acc) {
                    continue;
                }
                passed = test_shape(backend, n, explicit_acc) && passed;
            }
        }
        // Mat-vec (N=1) and mat-mat staging of non-contiguous operands.
        for (int n : {1, 37, 64}) {
            passed = test_permuted(backend, n) && passed;
        }
        ggml_backend_free(backend);
    }
    return !tested ? 77 : passed ? 0 : 1;
}
