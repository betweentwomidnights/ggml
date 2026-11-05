#include "common.h"

// One SIMDgroup per contiguous row. Subtract the maximum before exponentiating,
// and compute log-softmax directly so a very unlikely label still has finite loss.
kernel void kernel_cross_entropy_loss_f32(
        constant ggml_metal_kargs_cross_entropy_loss & args,
        device const float * logits,
        device const float * labels,
        device       float * losses,
        uint3 row [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    logits += int64_t(row.x)*args.nc;
    labels += int64_t(row.x)*args.nc;

    float max_logit = -INFINITY;
    for (int64_t c = lane; c < args.nc; c += N_SIMDWIDTH) {
        max_logit = max(max_logit, logits[c]);
    }
    max_logit = simd_max(max_logit);

    float sum_exp = 0.0f;
    for (int64_t c = lane; c < args.nc; c += N_SIMDWIDTH) {
        sum_exp += exp(logits[c] - max_logit);
    }
    const float log_sum = log(simd_sum(sum_exp));

    float loss = 0.0f;
    for (int64_t c = lane; c < args.nc; c += N_SIMDWIDTH) {
        loss += (logits[c] - max_logit - log_sum)*labels[c];
    }
    loss = -simd_sum(loss)/float(args.nr);
    if (lane == 0) {
        losses[row.x] = loss;
    }
}

kernel void kernel_cross_entropy_loss_reduce_f32(
        constant ggml_metal_kargs_cross_entropy_loss & args,
        device const float * losses,
        device       float * dst,
        ushort lane [[thread_index_in_simdgroup]]) {
    float loss = 0.0f;
    for (int64_t row = lane; row < args.nr; row += N_SIMDWIDTH) {
        loss += losses[row];
    }
    loss = simd_sum(loss);
    if (lane == 0) {
        dst[0] = loss;
    }
}

kernel void kernel_cross_entropy_loss_back_f32(
        constant ggml_metal_kargs_cross_entropy_loss & args,
        device const float * logits,
        device const float * labels,
        device       float * dst,
        device const float * grad,
        uint3 row [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    logits += int64_t(row.x)*args.nc;
    labels += int64_t(row.x)*args.nc;
    dst    += int64_t(row.x)*args.nc;

    float max_logit = -INFINITY;
    for (int64_t c = lane; c < args.nc; c += N_SIMDWIDTH) {
        max_logit = max(max_logit, logits[c]);
    }
    max_logit = simd_max(max_logit);

    float sum_exp = 0.0f;
    for (int64_t c = lane; c < args.nc; c += N_SIMDWIDTH) {
        sum_exp += exp(logits[c] - max_logit);
    }
    const float inv_sum = 1.0f/simd_sum(sum_exp);
    const float scale = grad[0]/float(args.nr);
    for (int64_t c = lane; c < args.nc; c += N_SIMDWIDTH) {
        dst[c] = (exp(logits[c] - max_logit)*inv_sum - labels[c])*scale;
    }
}

template<typename T>
kernel void kernel_soft_max(
        constant ggml_metal_kargs_soft_max & args,
        device const  char * src0,
        device const  char * src1,
        device const  char * src2,
        device        char * dst,
        threadgroup  float * buf [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint3  tptg[[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;
    const int32_t i01 = tgpig.x;

    const int32_t i13 = i03%args.ne13;
    const int32_t i12 = i02%args.ne12;
    const int32_t i11 = i01;

    device const float * psrc0 =                (device const float *) (src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);
    device const     T * pmask = src1 != src0 ? (device const T *    ) (src1 + i11*args.nb11 + i12*args.nb12 + i13*args.nb13) : nullptr;
    device const float * psrc2 = src2 != src0 ? (device const float *) (src2)                                                 : nullptr;
    device       float * pdst  =                (device       float *) (dst  + i01*args.nb1  + i02*args.nb2  + i03*args.nb3);

    float slope = 1.0f;

    // ALiBi
    if (args.max_bias > 0.0f) {
        const int32_t h = i02;

        const float base = h < args.n_head_log2 ? args.m0 : args.m1;
        const int   exp  = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

        slope = pow(base, exp);
    }

    // parallel max
    float lmax = psrc2 ? psrc2[i02] : -INFINITY;

    for (int i00 = tpitg.x; i00 < args.ne00; i00 += tptg.x) {
        lmax = MAX(lmax, psrc0[i00]*args.scale + (pmask ? slope*pmask[i00] : 0.0f));
    }

    // find the max value in the block
    float max_val = simd_max(lmax);
    if (tptg.x > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = -INFINITY;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = max_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = buf[tiisg];
        max_val = simd_max(max_val);
    }

    // parallel sum
    float lsum = 0.0f;
    for (int i00 = tpitg.x; i00 < args.ne00; i00 += tptg.x) {
        const float exp_psrc0 = exp((psrc0[i00]*args.scale + (pmask ? slope*pmask[i00] : 0.0f)) - max_val);
        lsum += exp_psrc0;
        pdst[i00] = exp_psrc0;
    }

    // This barrier fixes a failing test
    // ref: https://github.com/ggml-org/ggml/pull/621#discussion_r1425156335
    threadgroup_barrier(mem_flags::mem_none);

    float sum = simd_sum(lsum);

    if (tptg.x > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        sum = buf[tiisg];
        sum = simd_sum(sum);
    }

    if (psrc2) {
        sum += exp(psrc2[i02] - max_val);
    }

    const float inv_sum = 1.0f/sum;

    for (int i00 = tpitg.x; i00 < args.ne00; i00 += tptg.x) {
        pdst[i00] *= inv_sum;
    }
}

template<typename T>
kernel void kernel_soft_max_4(
        constant ggml_metal_kargs_soft_max & args,
        device const  char * src0,
        device const  char * src1,
        device const  char * src2,
        device        char * dst,
        threadgroup  float * buf [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint3  tptg[[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;
    const int32_t i01 = tgpig.x;

    const int32_t i13 = i03%args.ne13;
    const int32_t i12 = i02%args.ne12;
    const int32_t i11 = i01;

    device const float4 * psrc4 =                (device const float4 *) (src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);
    device const      T * pmask = src1 != src0 ? (device const T *     ) (src1 + i11*args.nb11 + i12*args.nb12 + i13*args.nb13) : nullptr;
    device const float *  psrc2 = src2 != src0 ? (device const float * ) (src2)                                                 : nullptr;
    device       float4 * pdst4 =                (device       float4 *) (dst  + i01*args.nb1  + i02*args.nb2  + i03*args.nb3);

    float slope = 1.0f;

    if (args.max_bias > 0.0f) {
        const int32_t h = i02;

        const float base = h < args.n_head_log2 ? args.m0 : args.m1;
        const int   exp  = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

        slope = pow(base, exp);
    }

    // parallel max
    float4 lmax4 = psrc2 ? psrc2[i02] : -INFINITY;

    for (int i00 = tpitg.x; i00 < args.ne00/4; i00 += tptg.x) {
        lmax4 = fmax(lmax4, psrc4[i00]*args.scale + (float4)((pmask ? slope*pmask[i00] : 0.0f)));
    }

    const float lmax = MAX(MAX(lmax4[0], lmax4[1]), MAX(lmax4[2], lmax4[3]));

    float max_val = simd_max(lmax);
    if (tptg.x > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = -INFINITY;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = max_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = buf[tiisg];
        max_val = simd_max(max_val);
    }

    // parallel sum
    float4 lsum4 = 0.0f;
    for (int i00 = tpitg.x; i00 < args.ne00/4; i00 += tptg.x) {
        const float4 exp_psrc4 = exp((psrc4[i00]*args.scale + (float4)((pmask ? slope*pmask[i00] : 0.0f))) - max_val);
        lsum4 += exp_psrc4;
        pdst4[i00] = exp_psrc4;
    }

    const float lsum = lsum4[0] + lsum4[1] + lsum4[2] + lsum4[3];

    // This barrier fixes a failing test
    // ref: https://github.com/ggml-org/ggml/pull/621#discussion_r1425156335
    threadgroup_barrier(mem_flags::mem_none);

    float sum = simd_sum(lsum);

    if (tptg.x > N_SIMDWIDTH) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        sum = buf[tiisg];
        sum = simd_sum(sum);
    }

    if (psrc2) {
        sum += exp(psrc2[i02] - max_val);
    }

    const float inv_sum = 1.0f/sum;

    for (int i00 = tpitg.x; i00 < args.ne00/4; i00 += tptg.x) {
        pdst4[i00] *= inv_sum;
    }
}

typedef decltype(kernel_soft_max<float>)    kernel_soft_max_t;
typedef decltype(kernel_soft_max_4<float4>) kernel_soft_max_4_t;

template [[host_name("kernel_soft_max_f16")]]   kernel kernel_soft_max_t   kernel_soft_max<half>;
template [[host_name("kernel_soft_max_f32")]]   kernel kernel_soft_max_t   kernel_soft_max<float>;
template [[host_name("kernel_soft_max_f16_4")]] kernel kernel_soft_max_4_t kernel_soft_max_4<half4>;
template [[host_name("kernel_soft_max_f32_4")]] kernel kernel_soft_max_4_t kernel_soft_max_4<float4>;

kernel void kernel_soft_max_back_f32(
        constant ggml_metal_kargs_soft_max_back & args,
        device const char * grad,
        device const char * y,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    device const float * grad_row = (device const float *)
        (grad + tgpig.x*args.nb01 + tgpig.y*args.nb02 + tgpig.z*args.nb03);
    device const float * y_row = (device const float *)
        (y + tgpig.x*args.nb11 + tgpig.y*args.nb12 + tgpig.z*args.nb13);
    device float * dst_row = (device float *)
        (dst + tgpig.x*args.nb1 + tgpig.y*args.nb2 + tgpig.z*args.nb3);

    float dot_yg = 0.0f;
    for (int i0 = tpitg.x; i0 < args.ne00; i0 += 32) {
        dot_yg += y_row[i0]*grad_row[i0];
    }
    dot_yg = simd_sum(dot_yg);

    for (int i0 = tpitg.x; i0 < args.ne00; i0 += 32) {
        dst_row[i0] = args.scale*(grad_row[i0] - dot_yg)*y_row[i0];
    }
}
