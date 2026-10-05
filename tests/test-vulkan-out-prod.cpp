#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

struct shape {
    int m, n, k, a2, a3, repeat2, repeat3;
};

// Use an independent scalar reference: the CPU backend does not support BF16
// OUT_PROD, so a backend-to-CPU comparison would silently skip these cases.
static bool check(ggml_backend_t backend, ggml_type type, shape s, bool padded_a, bool tb, int exponent) {
    ggml_context * ctx = ggml_init({1024 * 1024, nullptr, true});
    if (!ctx) return false;
    const int b2 = s.a2 * s.repeat2, b3 = s.a3 * s.repeat3;
    const int storage_m = s.m + (padded_a ? 3 : 0);
    ggml_tensor * as = ggml_new_tensor_4d(ctx, type, storage_m, s.k, s.a2, s.a3);
    ggml_tensor * bs = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, tb ? s.k : s.n, tb ? s.n : s.k, b2, b3);
    // OUT_PROD forbids transposed src0; a padded row view exercises legal
    // non-contiguous input strides without constructing an invalid graph.
    ggml_tensor * a = padded_a ? ggml_view_4d(ctx, as, s.m, s.k, s.a2, s.a3, as->nb[1], as->nb[2], as->nb[3], 0) : as;
    ggml_tensor * b = tb ? ggml_transpose(ctx, bs) : bs;
    ggml_tensor * dst = ggml_out_prod(ctx, a, b);
    // BF16 support applies to frozen src0 only; gradient and result stay F32.
    ggml_tensor invalid = *dst;
    invalid.type = GGML_TYPE_F16;
    ggml_tensor half_gradient = *b;
    half_gradient.type = GGML_TYPE_F16;
    const bool bad_output_supported = ggml_backend_supports_op(backend, &invalid);
    invalid = *dst;
    invalid.src[1] = &half_gradient;
    if (bad_output_supported || ggml_backend_supports_op(backend, &invalid)) {
        fprintf(stderr, "OUT_PROD incorrectly accepts a non-F32 gradient or result\n");
        ggml_free(ctx);
        return false;
    }
    if (!ggml_backend_supports_op(backend, dst)) {
        fprintf(stderr, "%s: %s OUT_PROD unsupported\n", ggml_backend_name(backend), ggml_type_name(type));
        ggml_free(ctx);
        return false;
    }
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, dst);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) { ggml_free(ctx); return false; }
    std::vector<float> av(ggml_nelements(as)), bv(ggml_nelements(bs)), actual(ggml_nelements(dst));
    std::vector<ggml_fp16_t> ah(av.size());
    std::vector<ggml_bf16_t> ab(av.size());
    for (size_t i = 0; i < av.size(); ++i) {
        // Include negative values and rounding to the stored input precision.
        av[i] = std::ldexp(float(int(i * 13 % 97) - 48) / 17.0f, exponent);
        if (type == GGML_TYPE_F16) {
            ah[i] = ggml_fp32_to_fp16(av[i]); av[i] = ggml_fp16_to_fp32(ah[i]);
        } else if (type == GGML_TYPE_BF16) {
            ab[i] = ggml_fp32_to_bf16(av[i]); av[i] = ggml_bf16_to_fp32(ab[i]);
        }
    }
    for (size_t i = 0; i < bv.size(); ++i) bv[i] = float(int(i * 7 % 41) - 20) / 19.0f;
    const void * bytes = type == GGML_TYPE_F32 ? static_cast<void *>(av.data()) :
                         type == GGML_TYPE_F16 ? static_cast<void *>(ah.data()) : static_cast<void *>(ab.data());
    ggml_backend_tensor_set(as, bytes, 0, ggml_nbytes(as));
    ggml_backend_tensor_set(bs, bv.data(), 0, ggml_nbytes(bs));
    bool ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    if (ok) {
        ggml_backend_tensor_get(dst, actual.data(), 0, ggml_nbytes(dst));
        for (int i3 = 0; i3 < b3; ++i3) for (int i2 = 0; i2 < b2; ++i2) {
            const size_t ao = size_t((i3 / s.repeat3) * s.a2 + i2 / s.repeat2) * storage_m * s.k;
            const size_t bo = size_t(i3 * b2 + i2) * s.n * s.k;
            const size_t d = size_t(i3 * b2 + i2) * s.m * s.n;
            for (int row = 0; row < s.n; ++row) for (int col = 0; col < s.m; ++col) {
                double expected = 0, magnitude = 0;
                for (int k = 0; k < s.k; ++k) {
                    const double product = double(av[ao + k * storage_m + col]) *
                                                  bv[bo + (tb ? row * s.k + k : k * s.n + row)];
                    expected += product; magnitude += std::fabs(product);
                }
                const float value = actual[d + row * s.m + col];
                // Scale with sum(abs(products)), allowing F32 accumulation error
                // even near cancellation; keep tiny BF16 values observable.
                if (!std::isfinite(value) || std::fabs(double(value) - expected) > 1e-5 * magnitude + 1e-36) {
                    if (ok) fprintf(stderr, "mismatch at [%d,%d,%d,%d]: expected=%g actual=%g bound=%g\n",
                                    col, row, i2, i3, expected, double(value), 1e-5 * magnitude + 1e-36);
                    ok = false;
                }
            }
        }
    }
    printf("%s type=%s shape=[%d,%d,%d] batches=[%d,%d] repeats=[%d,%d] padded_a=%d transpose_b=%d exponent=%d %s\n",
           ggml_backend_name(backend), ggml_type_name(type), s.m, s.n, s.k, s.a2, s.a3,
           s.repeat2, s.repeat3, padded_a, tb, exponent, ok ? "PASS" : "FAIL");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    int devices = 0, cases = 0, failures = 0;
    const shape shapes[] = {
        {1,1,1,1,1,1,1}, {16,16,16,1,1,1,1}, {65,17,23,1,1,1,1},
        {33,7,31,2,3,1,1}, {17,9,65,1,1,2,3}, {67,5,129,2,1,2,2},
        {1024,3,2048,1,1,1,1},
    };
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        if (strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(device)), "Vulkan") != 0) continue;
        if (argc > 1 && strcmp(ggml_backend_dev_name(device), argv[1]) != 0) continue;
        ++devices;
        ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
        if (!backend) return 1;
        for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16}) {
            for (shape s : shapes) for (bool ta : {false,true}) for (bool tb : {false,true}) {
                ++cases; failures += !check(backend, type, s, ta, tb, 0);
            }
        }
        // BF16's exponent range differs from F16: catch accidental narrowing
        // or interpreting uint16 storage as a numerical integer.
        for (int exponent : {-80,80}) for (bool ta : {false,true}) for (bool tb : {false,true}) {
            ++cases; failures += !check(backend, GGML_TYPE_BF16, {65,17,23,1,1,1,1}, ta, tb, exponent);
        }
        ggml_backend_free(backend);
    }
    printf("devices=%d cases=%d failures=%d\n", devices, cases, failures);
    if (!devices) return argc > 1 ? 1 : 77;
    return failures ? 1 : 0;
}
