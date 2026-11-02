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

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    bool passed = true, tested = false;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        // This regression targets CUDA dispatch and its CPU oracle. Default
        // accumulation is not a portable F32 guarantee for other backends.
        const std::string name = ggml_backend_dev_name(dev);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU && name.rfind("CUDA", 0) != 0) {
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
        // MMVF, MMF interior/boundaries, and the ordinary GEMM path.
        for (int n : {1, 5, 8, 16, 17, 25, 64}) {
            for (bool explicit_acc : {false, true}) {
                passed = test_shape(backend, n, explicit_acc) && passed;
            }
        }
        ggml_backend_free(backend);
    }
    return !tested ? 77 : passed ? 0 : 1;
}
