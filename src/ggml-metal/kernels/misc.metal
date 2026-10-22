#include "common.h"
#include "dequantize.h"

kernel void kernel_argmax_f32(
        constant ggml_metal_kargs_argmax & args,
        device   const char * src0,
        device         char * dst,
        threadgroup    char * shmem [[threadgroup(0)]],
        uint  tgpig[[threadgroup_position_in_grid]],
        uint  tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint    ntg[[threads_per_threadgroup]]) {
    device const float * x_row = (device const float *) ((device const char *) src0 + tgpig * args.nb01);

    float   lmax = -INFINITY;
    int32_t larg = -1;

    for (int i00 = tpitg; i00 < args.ne00; i00 += ntg) {
        if (x_row[i00] > lmax) {
            lmax = x_row[i00];
            larg = i00;
        }
    }

    // find the argmax value in the block
    float max_val = simd_max(lmax);
    int32_t arg_val = simd_max(select(-1, larg, lmax == max_val));

    device int32_t * dst_i32 = (device int32_t *) dst;

    threadgroup   float * shared_maxval = (threadgroup   float *) shmem;
    threadgroup int32_t * shared_argmax = (threadgroup int32_t *) shmem + N_SIMDWIDTH;

    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            shared_maxval[tiisg] = -INFINITY;
            shared_argmax[tiisg] = -1;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            shared_maxval[sgitg] = max_val;
            shared_argmax[sgitg] = arg_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = shared_maxval[tiisg];
        arg_val = shared_argmax[tiisg];

        float max_val_reduced   = simd_max(max_val);
        int32_t arg_val_reduced = simd_max(select(-1, arg_val, max_val == max_val_reduced));

        dst_i32[tgpig] = arg_val_reduced;

        return;
    }

    dst_i32[tgpig] = arg_val;
}

kernel void kernel_diag_f32(
        constant ggml_metal_kargs_diag & args,
        device   const char * src0,
        device         char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]]) {
    constexpr short NW = N_SIMDWIDTH;

    const int32_t i3 = tgpig.z;
    const int32_t i2 = tgpig.y;
    const int32_t i1 = tgpig.x;

    device const float * src0_ptr = (device const float *)(src0 +                i2*args.nb02 + i3*args.nb03);
    device       float * dst_ptr  = (device       float *)(dst  + i1*args.nb01 + i2*args.nb2  + i3*args.nb3);

    for (int i0 = tiitg; i0 < args.ne0; i0 += NW) {
        dst_ptr[i0] = i0 == i1 ? src0_ptr[i0] : 0.0f;
    }
}

kernel void kernel_roll_f32(
    constant ggml_metal_kargs_roll & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    device const float * src0_ptr = (device const float *) src0;
    device       float * dst_ptr  = (device       float *) dst;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        // apply shifts and wrap around
        int64_t i00 = i0 - args.s0;
        int64_t i01 = i1 - args.s1;
        int64_t i02 = i2 - args.s2;
        int64_t i03 = i3 - args.s3;

        if (i00 < 0) { i00 += args.ne00; } else if (i00 >= args.ne00) { i00 -= args.ne00; }
        if (i01 < 0) { i01 += args.ne01; } else if (i01 >= args.ne01) { i01 -= args.ne01; }
        if (i02 < 0) { i02 += args.ne02; } else if (i02 >= args.ne02) { i02 -= args.ne02; }
        if (i03 < 0) { i03 += args.ne03; } else if (i03 >= args.ne03) { i03 -= args.ne03; }

        int64_t src_idx = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00 + i00;
        int64_t dst_idx = i3 *args.ne2 *args.ne1 *args.ne0  + i2 *args.ne1 *args.ne0  + i1 *args.ne0  + i0;

        dst_ptr[dst_idx] = src0_ptr[src_idx];
    }
}

constant bool FC_pad_circular [[function_constant(FC_PAD + 0)]];

// circular means on a torus, so the coordinates wrap around
static inline int32_t wrap_around(int32_t coord, int32_t size) {
    return (coord + size) % size;
}

kernel void kernel_pad_f32(
    constant ggml_metal_kargs_pad & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {
    const int32_t i3 = tgpig.z;
    const int32_t i2 = tgpig.y;
    const int32_t k0 = tgpig.x/args.ne1;
    const int32_t i1 = tgpig.x - k0*args.ne1;

    const int32_t ne00 = args.ne00;
    const int32_t ne01 = args.ne01;
    const int32_t ne02 = args.ne02;
    const int32_t ne03 = args.ne03;

    int32_t i01 = i1 - args.lp1;
    int32_t i02 = i2 - args.lp2;
    int32_t i03 = i3 - args.lp3;

    if (FC_pad_circular) {
        i01 = wrap_around(i01, ne01);
        i02 = wrap_around(i02, ne02);
        i03 = wrap_around(i03, ne03);
    }

    device float * dst_ptr = (device float *) (dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1);

    // the row lies in the padded region, so no source row backs it
    if (i01 < 0 || i01 >= ne01 ||
        i02 < 0 || i02 >= ne02 ||
        i03 < 0 || i03 >= ne03) {
        for (int32_t l0 = 0; l0 < 1024; l0 += ntg.x) {
            const int32_t i0 = k0*1024 + tpitg.x + l0;
            if (i0 >= args.ne0) {
                break;
            }

            dst_ptr[i0] = 0.0f;
        }

        return;
    }

    device const char * src0_row = src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01;

    for (int32_t l0 = 0; l0 < 1024; l0 += ntg.x) {
        const int32_t i0 = k0*1024 + tpitg.x + l0;
        if (i0 >= args.ne0) {
            break;
        }

        int32_t i00 = i0 - args.lp0;

        if (FC_pad_circular) {
            i00 = wrap_around(i00, ne00);
        }

        dst_ptr[i0] = i00 >= 0 && i00 < ne00 ? *((device const float *) (src0_row + i00*args.nb00)) : 0.0f;
    }
}

// TODO: this is slow - optimize
kernel void kernel_pad_reflect_1d_f32(
    constant   ggml_metal_kargs_pad_reflect_1d & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3  tgpg[[threadgroups_per_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    const int64_t i03 = i3;
    const int64_t i02 = i2;
    const int64_t i01 = i1;

    device const float * src0_ptr = (device const float *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01);
    device       float * dst_ptr  = (device       float *) (dst  +  i3*args.nb3  +  i2*args.nb2  +  i1*args.nb1);

    if (i1 < args.ne01 && i2 < args.ne02 && i3 < args.ne03) {
        for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
            if (i0 < args.p0) {
                dst_ptr[i0] = src0_ptr[args.p0 - i0];
            } else if (i0 < args.ne0 - args.p1) {
                dst_ptr[i0] = src0_ptr[i0 - args.p0];
            } else {
                dst_ptr[i0] = src0_ptr[(args.ne0 - args.p1 - args.p0) - (args.p1 + 1 - (args.ne0 - i0)) - 1];
            }
        }
    }
}

kernel void kernel_arange_f32(
    constant   ggml_metal_kargs_arange & args,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    device float * dst_ptr = (device float *) dst;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        dst_ptr[i0] = args.start + args.step * i0;
    }
}

kernel void kernel_timestep_embedding_f32(
    constant  ggml_metal_kargs_timestep_embedding & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    int i = tgpig.x;
    device float * embed_data = (device float *)(dst + i*args.nb1);

    int half_ = args.dim / 2;
    for (int j = tpitg.x; j < half_; j += ntg.x) {
        float timestep = ((device float *)src0)[i];
        float freq = (float)exp(-log((float)args.max_period) * j / half_);
        float arg = timestep * freq;
        embed_data[j        ] = cos(arg);
        embed_data[j + half_] = sin(arg);
    }

    if (args.dim % 2 != 0 && tpitg.x == 0) {
        embed_data[2 * half_] = 0.f;
    }
}

kernel void kernel_opt_step_adamw_f32(
        constant    ggml_metal_kargs_opt_step_adamw & args,
        device       float * x,
        device const float * g,
        device       float * g_m,
        device       float * g_v,
        device const float * pars,
        uint        gid[[thread_position_in_grid]]) {

    if (gid >= args.np) {
        return;
    }

    const float alpha  = pars[0];
    const float beta1  = pars[1];
    const float beta2  = pars[2];
    const float eps    = pars[3];
    const float wd     = pars[4];
    const float beta1h = pars[5];
    const float beta2h = pars[6];

    const float gi = g[gid];
    const float gmi = g_m[gid] * beta1 +      gi * (1.0f - beta1);
    const float gvi = g_v[gid] * beta2 + gi * gi * (1.0f - beta2);

    g_m[gid] = gmi;
    g_v[gid] = gvi;

    const float mh =      gmi * beta1h;
    const float vh = sqrt(gvi * beta2h) + eps;

    x[gid] = x[gid] * (1.0f - alpha * wd) - alpha * mh / vh;
}

kernel void kernel_opt_step_sgd_f32(
        constant    ggml_metal_kargs_opt_step_sgd & args,
        device       float * x,
        device const float * g,
        device const float * pars,
        uint        gid[[thread_position_in_grid]]) {

    if (gid >= args.np) {
        return;
    }

    x[gid] = x[gid] * (1.0f - pars[0] * pars[1]) - pars[0] * g[gid];
}

template<typename T>
kernel void kernel_memset(
        constant ggml_metal_kargs_memset & args,
        device T * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = args.val;
}

typedef decltype(kernel_memset<int64_t>) kernel_memset_t;

template [[host_name("kernel_memset_i64")]] kernel kernel_memset_t kernel_memset<int64_t>;

constant short FC_count_equal_nsg [[function_constant(FC_COUNT_EQUAL + 0)]];

template<typename T>
kernel void kernel_count_equal(
        constant ggml_metal_kargs_count_equal & args,
        device   const char * src0,
        device   const char * src1,
        device   atomic_int * dst,
        threadgroup int32_t * shmem_i32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const short NSG = FC_count_equal_nsg;

    const int i3 = tgpig.z;
    const int i2 = tgpig.y;
    const int i1 = tgpig.x;

    if (i3 >= args.ne03 || i2 >= args.ne02 || i1 >= args.ne01) {
        return;
    }

    int sum = 0;

    device const char * base0 = src0 + i1*args.nb01 + i2*args.nb02 + i3*args.nb03;
    device const char * base1 = src1 + i1*args.nb11 + i2*args.nb12 + i3*args.nb13;

    for (int64_t i0 = tpitg.x; i0 < args.ne00; i0 += ntg.x) {
        const T v0 = *(device const T *)(base0 + i0*args.nb00);
        const T v1 = *(device const T *)(base1 + i0*args.nb10);
        sum += (v0 == v1);
    }

    sum = simd_sum(sum);

    if (tiisg == 0) {
        shmem_i32[sgitg] = sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        float v = 0.0f;
        if (tpitg.x < NSG) {
            v = shmem_i32[tpitg.x];
        }

        float total = simd_sum(v);
        if (tpitg.x == 0) {
            atomic_fetch_add_explicit(dst, (int32_t) total, memory_order_relaxed);
        }
    }
}

typedef decltype(kernel_count_equal<int32_t>) kernel_count_equal_t;

template [[host_name("kernel_count_equal_i32")]] kernel kernel_count_equal_t kernel_count_equal<int32_t>;

template <typename T>
kernel void kernel_snake(
        constant ggml_metal_kargs_snake & args,
        device const T     * x,
        device const float * a,
        device const float * inv_b,
        device       T     * dst,
        uint         tgpig [[threadgroup_position_in_grid]],
        uint         tpitg [[thread_position_in_threadgroup]],
        uint         ntg   [[threads_per_threadgroup]]) {

    const int idx = tgpig * ntg + tpitg;
    if (idx >= args.T * args.C) {
        return;
    }

    const int   c  = idx / args.T;  // x is [T, C], a / inv_b collapse to [1, C]
    const float xi = float(x[idx]);
    const float si = sin(a[c] * xi);
    dst[idx] = T(xi + si * si * inv_b[c]);
}

template [[host_name("kernel_snake_f32")]]  kernel void kernel_snake<float>(constant ggml_metal_kargs_snake &, device const float *, device const float *, device const float *, device float *, uint, uint, uint);
template [[host_name("kernel_snake_f16")]]  kernel void kernel_snake<half>(constant ggml_metal_kargs_snake &, device const half *, device const float *, device const float *, device half *, uint, uint, uint);
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_snake_bf16")]] kernel void kernel_snake<bfloat>(constant ggml_metal_kargs_snake &, device const bfloat *, device const float *, device const float *, device bfloat *, uint, uint, uint);
#endif

template<int N, typename src_t>
kernel void kernel_fwht(
        constant ggml_metal_kargs_fwht & args,
        device const src_t * src,
        device float * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort3  ntg[[threads_per_threadgroup]]) {

    constexpr int NW = N_SIMDWIDTH;
    constexpr int NE = N / NW;

    const float scale = 1.0f / sqrt((float) N);

    const int sg_per_tg = ntg.x / NW;
    const int64_t r = tgpig.x * sg_per_tg + sgitg;
    if (r >= args.nrows) {
        return;
    }

    src += r * N;
    dst += r * N;

    const int lane = tiisg;

    float reg[NE];
    for (int i = 0; i < NE; i++) {
        reg[i] = float(src[i*NW + lane])*scale;
    }
    for (int i = 1; i < NW; i *= 2) {
        for (int j = 0; j < NE; j++) {
            const float val = reg[j];
            const float val2 = simd_shuffle_xor(val, i);
            reg[j] = val2 - val + 2*((lane & i) == 0)*val;
        }
    }

    for (int i = NW; i < N; i *= 2) {
        const int step = i / NW;
        for (int j = 0; j < NE; j += (2 * step)) {
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k ];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

    for (int i = 0; i < NE; i++) {
        dst[i*NW + lane] = reg[i];
    }
}

// Wide blocks: one row per threadgroup instead of per simdgroup, so each thread keeps
// N/NT values rather than N/32. Butterflies below the simdgroup width still shuffle;
// those up to NT go through threadgroup memory; the rest stay in registers.
// TODO: try avoiding branch https://github.com/ggml-org/llama.cpp/pull/29094#discussion_r4049563223
// TODO: try to unroll loops
template<int N, int NT, typename src_t>
kernel void kernel_fwht_tg(
        constant ggml_metal_kargs_fwht & args,
        device const src_t * src,
        device float * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort3  ntg[[threads_per_threadgroup]]) {

    constexpr int NW = N_SIMDWIDTH;
    constexpr int NE = N / NT;

    threadgroup float shmem[N];

    const float scale = 1.0f / sqrt((float) N);

    const int64_t r = tgpig.x;
    if (r >= args.nrows) {
        return;
    }

    src += r * N;
    dst += r * N;

    const int tid = sgitg * NW + tiisg;

    float reg[NE];
    for (int i = 0; i < NE; i++) {
        reg[i] = float(src[i*NT + tid])*scale;
    }

    for (int i = 1; i < NW; i *= 2) {
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = simd_shuffle_xor(val, i);
            reg[j] = (tid & i) == 0 ? val2 + val : val2 - val;
        }
    }

    for (int i = NW; i < NT; i *= 2) {
        for (int j = 0; j < NE; j++) {
            shmem[j*NT + tid] = reg[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = shmem[j*NT + (tid ^ i)];
            reg[j] = (tid & i) == 0 ? val2 + val : val2 - val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int i = NT; i < N; i *= 2) {
        const int step = i / NT;
        for (int j = 0; j < NE; j += (2 * step)) {
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k ];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

    for (int i = 0; i < NE; i++) {
        dst[i*NT + tid] = reg[i];
    }
}

typedef decltype(kernel_fwht<64, float>) kernel_fwht_f32_t;
typedef decltype(kernel_fwht<64, half>)  kernel_fwht_f16_t;

template [[host_name("kernel_fwht_f32_64")]]  kernel kernel_fwht_f32_t kernel_fwht<64,  float>;
template [[host_name("kernel_fwht_f32_128")]] kernel kernel_fwht_f32_t kernel_fwht<128, float>;
template [[host_name("kernel_fwht_f32_256")]] kernel kernel_fwht_f32_t kernel_fwht<256, float>;

template [[host_name("kernel_fwht_f16_64")]]  kernel kernel_fwht_f16_t kernel_fwht<64,  half>;
template [[host_name("kernel_fwht_f16_128")]] kernel kernel_fwht_f16_t kernel_fwht<128, half>;
template [[host_name("kernel_fwht_f16_256")]] kernel kernel_fwht_f16_t kernel_fwht<256, half>;

template [[host_name("kernel_fwht_f32_512")]]  kernel kernel_fwht_f32_t kernel_fwht_tg<512,  GGML_METAL_FWHT_TG_NT, float>;
template [[host_name("kernel_fwht_f32_1024")]] kernel kernel_fwht_f32_t kernel_fwht_tg<1024, GGML_METAL_FWHT_TG_NT, float>;
template [[host_name("kernel_fwht_f32_2048")]] kernel kernel_fwht_f32_t kernel_fwht_tg<2048, GGML_METAL_FWHT_TG_NT, float>;
template [[host_name("kernel_fwht_f32_4096")]] kernel kernel_fwht_f32_t kernel_fwht_tg<4096, GGML_METAL_FWHT_TG_NT, float>;
template [[host_name("kernel_fwht_f32_8192")]] kernel kernel_fwht_f32_t kernel_fwht_tg<8192, GGML_METAL_FWHT_TG_NT, float>;

template [[host_name("kernel_fwht_f16_512")]]  kernel kernel_fwht_f16_t kernel_fwht_tg<512,  GGML_METAL_FWHT_TG_NT, half>;
template [[host_name("kernel_fwht_f16_1024")]] kernel kernel_fwht_f16_t kernel_fwht_tg<1024, GGML_METAL_FWHT_TG_NT, half>;
template [[host_name("kernel_fwht_f16_2048")]] kernel kernel_fwht_f16_t kernel_fwht_tg<2048, GGML_METAL_FWHT_TG_NT, half>;
template [[host_name("kernel_fwht_f16_4096")]] kernel kernel_fwht_f16_t kernel_fwht_tg<4096, GGML_METAL_FWHT_TG_NT, half>;
template [[host_name("kernel_fwht_f16_8192")]] kernel kernel_fwht_f16_t kernel_fwht_tg<8192, GGML_METAL_FWHT_TG_NT, half>;

constant int FC_dsv4_hc_n_hc [[function_constant(FC_DSV4_HC + 0)]];

kernel void kernel_dsv4_hc_comb_f32(
        constant ggml_metal_kargs_dsv4_hc_comb & args,
        device const char * mixes,
        device const char * scale,
        device const char * base,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    constexpr ushort hc = 4;
    constexpr ushort comb_offset = 2*hc;

    const int it = tgpig.x*ntg.y + sgitg;
    if (it >= args.n_tokens) {
        return;
    }

    float scale_lane = 0.0f;
    if (tiisg == 0) {
        scale_lane = *(device const float *) (scale + 2*args.nb_s0);
    }
    const float scale_comb = simd_shuffle(scale_lane, 0);

    float v = 0.0f;
    if (tiisg < hc*hc) {
        v = *(device const float *) (mixes + (comb_offset + tiisg)*args.nb_m0 + it*args.nb_m1)*scale_comb
          + *(device const float *) (base   + (comb_offset + tiisg)*args.nb_b0);
    }

    // Softmax across destinations (the four contiguous lanes for each source).
    float vmax = max(v, simd_shuffle_xor(v, 1));
    vmax = max(vmax, simd_shuffle_xor(vmax, 2));
    v = exp(v - vmax);

    float sum = v + simd_shuffle_xor(v, 1);
    sum += simd_shuffle_xor(sum, 2);
    v = v/sum + args.eps;

    // Normalize columns: equal destination indices are four lanes apart.
    sum = v + simd_shuffle_xor(v, 4);
    sum += simd_shuffle_xor(sum, 8);
    v /= sum + args.eps;

    for (int i = 1; i < args.n_iter; ++i) {
        sum = v + simd_shuffle_xor(v, 1);
        sum += simd_shuffle_xor(sum, 2);
        v /= sum + args.eps;

        sum = v + simd_shuffle_xor(v, 4);
        sum += simd_shuffle_xor(sum, 8);
        v /= sum + args.eps;
    }

    if (tiisg < hc*hc) {
        const ushort idst = tiisg & 3;
        const ushort isrc = tiisg >> 2;
        *(device float *) (dst + idst*args.nb_d0 + isrc*args.nb_d1 + it*args.nb_d2) = v;
    }
}

kernel void kernel_dsv4_hc_pre_f32(
        constant ggml_metal_kargs_dsv4_hc_pre & args,
        device const char * x,
        device const char * weights,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int it = tgpig.y;
    const int i0 = ((int) tgpig.x*ntg.y + sgitg)*32 + tiisg;

    if (i0 >= args.n_embd) {
        return;
    }

    device const char * xb = x + i0*args.nb_x0 + it*args.nb_x2;
    float result = 0.0f;
    FOR_UNROLL (int ih = 0; ih < FC_dsv4_hc_n_hc; ++ih) {
        const float xv = *(device const float *) (xb + ih*args.nb_x1);
        const float wv = *(device const float *) (weights + ih*args.nb_w0 + it*args.nb_w1);
        result = fma(xv, wv, result);
    }

    *(device float *) (dst + i0*args.nb_d0 + it*args.nb_d1) = args.scale*result;
}

kernel void kernel_dsv4_hc_pre_gated_f32(
        constant ggml_metal_kargs_dsv4_hc_pre & args,
        device const char * x,
        device const char * gate,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int it = tgpig.y;
    const int i0 = ((int) tgpig.x*ntg.y + sgitg)*32 + tiisg;

    if (i0 >= args.n_embd) {
        return;
    }

    device const char * xb = x    + i0*args.nb_x0 + it*args.nb_x2;
    device const char * gb = gate + i0*args.nb_w0 + it*args.nb_w2;
    float result = 0.0f;
    FOR_UNROLL (int ih = 0; ih < FC_dsv4_hc_n_hc; ++ih) {
        const float g  = 1.0f/(1.0f + exp(-*(device const float *) (gb + ih*args.nb_w1)));
        const float xv = *(device const float *) (xb + ih*args.nb_x1);
        result = fma(xv, g, result);
    }

    *(device float *) (dst + i0*args.nb_d0 + it*args.nb_d1) = args.scale*result;
}

kernel void kernel_dsv4_hc_post_nocomb_f32(
        constant ggml_metal_kargs_dsv4_hc_post & args,
        device const char * x,
        device const char * residual,
        device const char * post,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    constexpr ushort hc = 4;

    const int it = tgpig.y;
    const int i0 = ((int) tgpig.x*ntg.y + sgitg)*32 + tiisg;

    float post_lane = 0.0f;
    if (tiisg < hc) {
        post_lane = *(device const float *) (post + tiisg*args.nb_p0 + it*args.nb_p1);
    }

    float post_reg[hc];
    FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
        post_reg[idst] = simd_shuffle(post_lane, idst);
    }

    if (i0 >= args.n_embd) {
        return;
    }

    const float xv = *(device const float *) (x + i0*args.nb_x0 + it*args.nb_x1);
    device const char * rb = residual + i0*args.nb_r0 + it*args.nb_r2;
    FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
        const float rv = *(device const float *) (rb + idst*args.nb_r1);
        *(device float *) (dst + i0*args.nb_d0 + idst*args.nb_d1 + it*args.nb_d2) = xv*post_reg[idst] + rv;
    }
}

kernel void kernel_dsv4_hc_post_f32(
        constant ggml_metal_kargs_dsv4_hc_post & args,
        device const char * x,
        device const char * residual,
        device const char * post,
        device const char * comb,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    constexpr ushort hc = 4;

    const int it = tgpig.y;
    const int i0 = ((int) tgpig.x*ntg.y + sgitg)*32 + tiisg;

    float coeff_lane = 0.0f;
    if (tiisg < hc) {
        coeff_lane = *(device const float *) (post + tiisg*args.nb_p0 + it*args.nb_p1);
    } else if (tiisg < hc + hc*hc) {
        const ushort idx  = tiisg - hc;
        const ushort idst = idx & 3;
        const ushort isrc = idx >> 2;
        coeff_lane = *(device const float *) (comb + idst*args.nb_c0 + isrc*args.nb_c1 + it*args.nb_c2);
    }

    float post_reg[hc];
    float comb_reg[hc][hc];
    FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
        post_reg[idst] = simd_shuffle(coeff_lane, idst);
    }
    FOR_UNROLL (ushort isrc = 0; isrc < hc; ++isrc) {
        FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
            comb_reg[isrc][idst] = simd_shuffle(coeff_lane, hc + idst + hc*isrc);
        }
    }

    if (i0 >= args.n_embd) {
        return;
    }

    const float xv = *(device const float *) (x + i0*args.nb_x0 + it*args.nb_x1);
    float result[hc];
    FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
        result[idst] = xv*post_reg[idst];
    }

    device const char * rb = residual + i0*args.nb_r0 + it*args.nb_r2;
    FOR_UNROLL (ushort isrc = 0; isrc < hc; ++isrc) {
        const float rv = *(device const float *) (rb + isrc*args.nb_r1);
        FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
            result[idst] = fma(rv, comb_reg[isrc][idst], result[idst]);
        }
    }

    FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
        *(device float *) (dst + i0*args.nb_d0 + idst*args.nb_d1 + it*args.nb_d2) = result[idst];
    }
}

kernel void kernel_repeat_back_f32(
        constant ggml_metal_kargs_repeat & args,
        device const char * src0,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i3 = tgpig.z;
    const int i2 = tgpig.y;
    const int i1 = tgpig.x;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        float sum = 0.0f;

        for (int i03 = i3; i03 < args.ne03; i03 += args.ne3) {
            for (int i02 = i2; i02 < args.ne02; i02 += args.ne2) {
                for (int i01 = i1; i01 < args.ne01; i01 += args.ne1) {
                    for (int i00 = i0; i00 < args.ne00; i00 += args.ne0) {
                        sum += *((device const float *) (src0 + i03*args.nb03 + i02*args.nb02 +
                                                        i01*args.nb01 + i00*args.nb00));
                    }
                }
            }
        }

        *((device float *) (dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0)) = sum;
    }
}

template<typename T0>
kernel void kernel_out_prod(
        constant ggml_metal_kargs_out_prod & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr int tile_x = 32;
    constexpr int tile_y = 16;
    constexpr int tile_k = 16;
    threadgroup float tile_a[tile_x][tile_k + 1];
    threadgroup float tile_b[tile_y][tile_k + 1];
    threadgroup float tile_c[tile_x][tile_y + 1];

    const int base_i0 = tgpig.x*tile_x;
    const int base_i1 = tgpig.y*tile_y;

    const int i2 = tgpig.z % args.ne2;
    const int i3 = tgpig.z / args.ne2;
    const int r2 = args.ne2 / args.ne02;
    const int r3 = args.ne3 / args.ne03;
    const int i02 = i2 / r2;
    const int i03 = i3 / r3;

    const int tile_i0 = (sgitg & 3)*8;
    const int tile_i1 = (sgitg >> 2)*8;
    simdgroup_float8x8 sum = make_filled_simdgroup_matrix<float, 8>(0.0f);

    for (int k0 = 0; k0 < args.ne01; k0 += tile_k) {
        // Eight simdgroups cooperatively stage a 32x16 A tile and a 16x16 B
        // tile. The asymmetric shape increases reuse for the typically large
        // first output dimension of LoRA weight gradients.
        for (int load = tiitg; load < tile_x*tile_k; load += 256) {
            const int ti = load / tile_k;
            const int tk = load % tile_k;
            const int k = k0 + tk;
            const int ai0 = base_i0 + ti;

            if (ai0 < args.ne0 && k < args.ne01) {
                tile_a[ti][tk] = float(*((device const T0 *)
                    (src0 + i03*args.nb03 + i02*args.nb02 + k*args.nb01 + ai0*args.nb00)));
            } else {
                tile_a[ti][tk] = 0.0f;
            }
        }

        for (int load = tiitg; load < tile_y*tile_k; load += 256) {
            const int ti = load / tile_k;
            const int tk = load % tile_k;
            const int k = k0 + tk;
            const int bi1 = base_i1 + ti;

            if (bi1 < args.ne1 && k < args.ne01) {
                tile_b[ti][tk] = *((device const float *)
                    (src1 + i3*args.nb13 + i2*args.nb12 + k*args.nb11 + bi1*args.nb10));
            } else {
                tile_b[ti][tk] = 0.0f;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int kk = 0; kk < tile_k; kk += 8) {
            simdgroup_float8x8 ma;
            simdgroup_float8x8 mb;
            simdgroup_load(ma, &tile_a[tile_i0][kk], tile_k + 1);
            simdgroup_load(mb, &tile_b[tile_i1][kk], tile_k + 1, 0, true);
            simdgroup_multiply_accumulate(sum, ma, mb, sum);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(sum, &tile_c[tile_i0][tile_i1], tile_y + 1);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int store = tiitg; store < tile_x*tile_y; store += 256) {
        const int li0 = store / tile_y;
        const int li1 = store % tile_y;
        const int i0 = base_i0 + li0;
        const int i1 = base_i1 + li1;
        if (i0 < args.ne0 && i1 < args.ne1) {
            *((device float *) (dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0)) = tile_c[li0][li1];
        }
    }
}

typedef decltype(kernel_out_prod<float>) kernel_out_prod_t;

template [[host_name("kernel_out_prod_f32_f32")]] kernel kernel_out_prod_t kernel_out_prod<float>;
template [[host_name("kernel_out_prod_f16_f32")]] kernel kernel_out_prod_t kernel_out_prod<half>;

// OUT_PROD with a quantized src0, which is what the backward of mul_mat needs to
// train a LoRA on a quantized base. Same structure as kernel_out_prod above —
// keep the two in sync — with the A tile dequantized inline instead of read as a
// typed element.
//
// The element mapping is the part worth reading twice. Metal's dequantize helpers
// emit 16 values at a time and take a *group* index within the block, the
// convention kernel_get_rows_q spells as dequantize_func(psrc + ind/nl, ind%nl),
// where ind counts 16-element groups along ne00. Those 16 values are consecutive
// along ne00, which is the axis tile_a's first index walks, so a work item here
// stages a whole 16-wide run of one src0 row rather than a single element. Doing
// it per element instead would repeat the block's scale unpack sixteen times, and
// that unpack — not the element arithmetic — is what a k-quant costs.
//
// That 16-wide granularity is also why tile_k is 32 here and 16 there. A tile is
// tile_x*tile_k/16 dequantize calls, so at tile_k = 16 only 32 of the 256 threads
// stage A and the rest wait at the barrier; 32 puts 64 to work and measured 7%
// faster end to end. 64 measured identical to 32 and costs 14.7 KiB of
// threadgroup memory against 8.5 KiB, so it is not worth the occupancy.
template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread float4x4 &)>
kernel void kernel_out_prod_q(
        constant ggml_metal_kargs_out_prod & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr int tile_x = 32;
    constexpr int tile_y = 16;
    constexpr int tile_k = 32;
    constexpr int run_a  = 16;             // elements one dequantize_func call emits
    threadgroup float tile_a[tile_x][tile_k + 1];
    threadgroup float tile_b[tile_y][tile_k + 1];
    threadgroup float tile_c[tile_x][tile_y + 1];

    const int base_i0 = tgpig.x*tile_x;
    const int base_i1 = tgpig.y*tile_y;

    const int i2 = tgpig.z % args.ne2;
    const int i3 = tgpig.z / args.ne2;
    const int r2 = args.ne2 / args.ne02;
    const int r3 = args.ne3 / args.ne03;
    const int i02 = i2 / r2;
    const int i03 = i3 / r3;

    const int tile_i0 = (sgitg & 3)*8;
    const int tile_i1 = (sgitg >> 2)*8;
    simdgroup_float8x8 sum = make_filled_simdgroup_matrix<float, 8>(0.0f);

    for (int k0 = 0; k0 < args.ne01; k0 += tile_k) {
        for (int load = tiitg; load < (tile_x/run_a)*tile_k; load += 256) {
            const int tk  = load / (tile_x/run_a);
            const int ti  = (load % (tile_x/run_a))*run_a;
            const int k   = k0 + tk;
            const int ai0 = base_i0 + ti;

            if (ai0 < args.ne0 && k < args.ne01) {
                device const block_q * prow = (device const block_q *)
                    (src0 + i03*args.nb03 + i02*args.nb02 + k*args.nb01);

                const int ig = ai0/run_a;  // 16-element group index within the row

                float4x4 temp;
                dequantize_func(prow + ig/nl, ig%nl, temp);

                for (short j = 0; j < run_a; ++j) {
                    tile_a[ti + j][tk] = ai0 + j < args.ne0 ? temp[j/4][j%4] : 0.0f;
                }
            } else {
                for (short j = 0; j < run_a; ++j) {
                    tile_a[ti + j][tk] = 0.0f;
                }
            }
        }

        for (int load = tiitg; load < tile_y*tile_k; load += 256) {
            const int ti = load / tile_k;
            const int tk = load % tile_k;
            const int k = k0 + tk;
            const int bi1 = base_i1 + ti;

            if (bi1 < args.ne1 && k < args.ne01) {
                tile_b[ti][tk] = *((device const float *)
                    (src1 + i3*args.nb13 + i2*args.nb12 + k*args.nb11 + bi1*args.nb10));
            } else {
                tile_b[ti][tk] = 0.0f;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int kk = 0; kk < tile_k; kk += 8) {
            simdgroup_float8x8 ma;
            simdgroup_float8x8 mb;
            simdgroup_load(ma, &tile_a[tile_i0][kk], tile_k + 1);
            simdgroup_load(mb, &tile_b[tile_i1][kk], tile_k + 1, 0, true);
            simdgroup_multiply_accumulate(sum, ma, mb, sum);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(sum, &tile_c[tile_i0][tile_i1], tile_y + 1);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int store = tiitg; store < tile_x*tile_y; store += 256) {
        const int li0 = store / tile_y;
        const int li1 = store % tile_y;
        const int i0 = base_i0 + li0;
        const int i1 = base_i1 + li1;
        if (i0 < args.ne0 && i1 < args.ne1) {
            *((device float *) (dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0)) = tile_c[li0][li1];
        }
    }
}

typedef decltype(kernel_out_prod_q<block_q8_0, 2, dequantize_q8_0>) kernel_out_prod_q_t;

// Only the types a q4_k_m / q5_k_m / q8_0 mix actually produces. The rest of the
// type table is deliberately left out until someone validates it.
template [[host_name("kernel_out_prod_q8_0_f32")]] kernel kernel_out_prod_q_t kernel_out_prod_q<block_q8_0, 2,     dequantize_q8_0>;
template [[host_name("kernel_out_prod_q4_K_f32")]] kernel kernel_out_prod_q_t kernel_out_prod_q<block_q4_K, QK_NL, dequantize_q4_K>;
template [[host_name("kernel_out_prod_q5_K_f32")]] kernel kernel_out_prod_q_t kernel_out_prod_q<block_q5_K, QK_NL, dequantize_q5_K>;
template [[host_name("kernel_out_prod_q6_K_f32")]] kernel kernel_out_prod_q_t kernel_out_prod_q<block_q6_K, QK_NL, dequantize_q6_K>;

kernel void kernel_rms_norm_back_f32(
        constant ggml_metal_kargs_norm & args,
        device const char * grad,
        device const char * x,
        device       char * dst,
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    if (sgitg == 0) {
        shmem_f32[tiisg]      = 0.0f;
        shmem_f32[32 + tiisg] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int i1 = tgpig.x;
    const int i2 = tgpig.y;
    const int i3 = tgpig.z;

    device const float * grad_row = (device const float *)
        (grad + i3*args.nbf3[0] + i2*args.nbf2[0] + i1*args.nbf1[0]);
    device const float * x_row = (device const float *)
        (x + i3*args.nbf3[1] + i2*args.nbf2[1] + i1*args.nbf1[1]);
    device float * dst_row = (device float *)
        (dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1);

    float sum_xx = 0.0f;
    float sum_xg = 0.0f;
    for (int i0 = tpitg.x; i0 < args.ne00; i0 += ntg.x) {
        const float xf = x_row[i0];
        sum_xx += xf*xf;
        sum_xg += xf*grad_row[i0];
    }

    sum_xx = simd_sum(sum_xx);
    sum_xg = simd_sum(sum_xg);

    if (tiisg == 0) {
        shmem_f32[sgitg]      = sum_xx;
        shmem_f32[32 + sgitg] = sum_xg;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    sum_xx = simd_sum(shmem_f32[tiisg]);
    sum_xg = simd_sum(shmem_f32[32 + tiisg]);

    const float mean_eps = sum_xx/args.ne00 + args.eps;
    const float rrms = 1.0f/sqrt(mean_eps);
    const float scale_x = -sum_xg/(sum_xx + args.ne00*args.eps);

    for (int i0 = tpitg.x; i0 < args.ne00; i0 += ntg.x) {
        dst_row[i0] = (grad_row[i0] + x_row[i0]*scale_x)*rrms;
    }
}
