// With GGML_CUDA_TF32=0, explicit F32 precision must preserve F32 operands,
// including small-N MMF shapes. CTest supplies the cuBLAS TF32 opt-out.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static bool test_shape(ggml_backend_t backend, int n) {
    const int k = 512, m = 64;
    std::vector<float> a(k * m), b(k * n), actual(m * n);
    for (size_t i = 0; i < a.size(); ++i) a[i] = 0.3f * std::sin(float(i) * 0.137f) + 0.00013f;
    for (size_t i = 0; i < b.size(); ++i) b[i] = 0.2f * std::cos(float(i) * 0.171f) - 0.00017f;
    auto * ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
    if (!ctx) return false;
    auto * ta = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
    auto * tb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    auto * result = ggml_mul_mat(ctx, ta, tb);
    ggml_mul_mat_set_prec(result, GGML_PREC_F32);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) { ggml_free(ctx); return false; }
    ggml_backend_tensor_set(ta, a.data(), 0, a.size() * sizeof(float));
    ggml_backend_tensor_set(tb, b.data(), 0, b.size() * sizeof(float));
    bool passed = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    ggml_backend_tensor_get(result, actual.data(), 0, actual.size() * sizeof(float));
    double squared_error = 0, squared_ref = 0, max_abs = 0;
    for (int col = 0; col < n; ++col) for (int row = 0; row < m; ++row) {
        double expected = 0;
        for (int i = 0; i < k; ++i) expected += double(a[row * k + i]) * b[col * k + i];
        const double delta = actual[col * m + row] - expected;
        passed = passed && std::isfinite(actual[col * m + row]);
        squared_error += delta * delta;
        squared_ref += expected * expected;
        max_abs = std::fmax(max_abs, std::fabs(delta));
    }
    const double relative_l2 = std::sqrt(squared_error / squared_ref);
    passed = passed && relative_l2 < 1e-5 && max_abs < 1e-5;
    std::printf("%s K=%d M=%d N=%d: rel_l2=%.8g max_abs=%.8g %s\n",
                ggml_backend_name(backend), k, m, n, relative_l2, max_abs, passed ? "PASS" : "FAIL");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return passed;
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    bool passed = true, tested = false;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto device = ggml_backend_dev_get(i);
        if (argc > 1 && std::string(ggml_backend_dev_name(device)) != argv[1]) continue;
        auto backend = ggml_backend_dev_init(device, nullptr);
        if (!backend) continue;
        tested = true;
        for (int n : {1, 5, 25, 64}) passed = test_shape(backend, n) && passed;
        ggml_backend_free(backend);
    }
    return !tested ? 77 : passed ? 0 : 1;
}
