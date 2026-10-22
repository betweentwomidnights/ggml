// CONCAT must slice the logical upstream gradient even when a downstream
// transpose/permute makes its innermost stride larger than one element.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

static void check(ggml_backend_t backend, int dim, std::array<int, 4> axes, int trainable) {
    ggml_context * data = ggml_init({ggml_tensor_overhead()*8, nullptr, true});
    ggml_context * ctx = ggml_init({1024*1024, nullptr, true});
    ggml_backend_buffer_t buffer = nullptr;
    ggml_gallocr_t allocator = nullptr;
    auto cleanup = [&] {
        if (allocator) ggml_gallocr_free(allocator);
        if (buffer) ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_free(data);
    };
    try {
        const std::array<int64_t, 4> na = {3, 4, 2, 2};
        auto nb = na; nb[dim] += 1;
        auto * a = ggml_new_tensor(data, GGML_TYPE_F32, 4, na.data());
        auto * b = ggml_new_tensor(data, GGML_TYPE_F32, 4, nb.data());
        ggml_set_input(a); ggml_set_input(b);
        if (trainable & 1) ggml_set_param(a);
        if (trainable & 2) ggml_set_param(b);
        auto * cat = ggml_concat(ctx, a, b, dim);
        auto * perm = ggml_permute(ctx, cat, axes[0], axes[1], axes[2], axes[3]);
        auto * weight = ggml_new_tensor(data, GGML_TYPE_F32, 4, perm->ne);
        ggml_set_input(weight);
        auto * loss = ggml_sum(ctx, ggml_mul(ctx, ggml_cont(ctx, perm), weight));
        ggml_set_loss(loss);
        ggml_set_output(loss);
        auto * graph = ggml_new_graph_custom(ctx, 256, true);
        ggml_build_forward_expand(graph, loss);
        ggml_build_backward_expand(ctx, graph, nullptr);
        for (auto * tensor : {a, b}) {
            auto * grad = ggml_graph_get_grad(graph, tensor);
            if (grad) ggml_set_output(grad);
        }
        buffer = ggml_backend_alloc_ctx_tensors(data, backend);
        allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!buffer || !ggml_gallocr_alloc_graph(allocator, graph)) throw std::runtime_error("allocation failed");
        std::vector<float> av(ggml_nelements(a), 0.25f), bv(ggml_nelements(b), -0.5f);
        std::vector<float> weights(ggml_nelements(weight));
        for (size_t i = 0; i < weights.size(); ++i) weights[i] = float(int(i % 29) - 14) / 17;
        ggml_backend_tensor_set(a, av.data(), 0, av.size()*sizeof(float));
        ggml_backend_tensor_set(b, bv.data(), 0, bv.size()*sizeof(float));
        ggml_backend_tensor_set(weight, weights.data(), 0, weights.size()*sizeof(float));
        ggml_graph_reset(graph);
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("compute failed");
        for (int which = 0; which < 2; ++which) {
            auto * source = which == 0 ? a : b;
            auto * grad = ggml_graph_get_grad(graph, source);
            if (!(trainable & (1 << which))) {
                if (grad) throw std::runtime_error("frozen source has a gradient");
                continue;
            }
            if (!grad) throw std::runtime_error("missing source gradient");
            std::vector<float> actual(ggml_nelements(source));
            ggml_backend_tensor_get(grad, actual.data(), 0, actual.size()*sizeof(float));
            for (int64_t i = 0; i < ggml_nelements(source); ++i) {
                int64_t remain = i;
                std::array<int64_t, 4> coordinate;
                for (int axis = 0; axis < 4; ++axis) {
                    coordinate[axis] = remain % source->ne[axis];
                    remain /= source->ne[axis];
                }
                if (which == 1) coordinate[dim] += na[dim];
                int64_t offset = 0;
                // ggml_permute maps input axis j to output axis axes[j].
                for (int axis = 0; axis < 4; ++axis) offset += coordinate[axis] * weight->nb[axes[axis]]/sizeof(float);
                if (!std::isfinite(actual[i]) || std::abs(actual[i]-weights[offset]) > 1e-6f) {
                    std::fprintf(stderr, "dim=%d axes=%d%d%d%d trainable=%d source=%d index=%lld got=%g expected=%g\n",
                        dim, axes[0], axes[1], axes[2], axes[3], trainable, which, (long long)i, actual[i], weights[offset]);
                    throw std::runtime_error("incorrect concatenation gradient");
                }
            }
        }
    } catch (...) { cleanup(); throw; }
    cleanup();
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    const char * name = argc > 1 ? argv[1] : "CPU";
    ggml_backend_dev_t device = ggml_backend_dev_by_name(name);
    if (!device) { std::fprintf(stderr, "device unavailable: %s\n", name); return 1; }
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    if (!backend) return 1;
    int result = 0;
    try {
        for (int dim = 0; dim < 4; ++dim) {
            for (const auto axes : {std::array<int, 4>{0, 1, 2, 3}, {1, 0, 2, 3}, {2, 0, 3, 1}}) {
                for (int trainable : {1, 2, 3}) check(backend, dim, axes, trainable);
            }
        }
        std::printf("PASS %s: 36 concat backward cases (contiguous, transposed, permuted; each/both sources)\n", name);
    } catch (const std::exception & error) { std::fprintf(stderr, "%s\n", error.what()); result = 1; }
    ggml_backend_free(backend);
    return result;
}
