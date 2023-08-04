#import "ggml-metal.h"

#import "ggml.h"

#import <Foundation/Foundation.h>

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#ifdef GGML_METAL_NDEBUG
#define metal_printf(...)
#else
#define metal_printf(...) fprintf(stderr, __VA_ARGS__)
#endif

#define UNUSED(x) (void)(x)

struct ggml_metal_buffer {
    const char * name;

    void   * data;
    size_t   size;

    id<MTLBuffer> metal;
};

struct ggml_metal_context {
    int n_cb;

    float * logits;

    id<MTLDevice>       device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary>      library;

    int n_buffers;
    struct ggml_metal_buffer buffers[GGML_METAL_MAX_BUFFERS];

    int concur_list[GGML_MAX_NODES];
    int concur_list_len;

    // custom kernels
#define GGML_METAL_DECL_KERNEL(name) \
    id<MTLFunction>             function_##name; \
    id<MTLComputePipelineState> pipeline_##name

    GGML_METAL_DECL_KERNEL(add);
    GGML_METAL_DECL_KERNEL(add_row); // TODO: avoid this extra kernel, instead extend the "add" kernel to support broadcast
    GGML_METAL_DECL_KERNEL(mul);
    GGML_METAL_DECL_KERNEL(mul_row); // TODO: avoid this extra kernel, instead extend the "mul" kernel to support broadcast
    GGML_METAL_DECL_KERNEL(scale);
    GGML_METAL_DECL_KERNEL(silu);
    GGML_METAL_DECL_KERNEL(relu);
    GGML_METAL_DECL_KERNEL(gelu);
    GGML_METAL_DECL_KERNEL(soft_max);
    GGML_METAL_DECL_KERNEL(diag_mask_inf);
    GGML_METAL_DECL_KERNEL(get_rows_f16);
    GGML_METAL_DECL_KERNEL(get_rows_q4_0);
    GGML_METAL_DECL_KERNEL(get_rows_q4_1);
    GGML_METAL_DECL_KERNEL(get_rows_q2_K);
    GGML_METAL_DECL_KERNEL(get_rows_q3_K);
    GGML_METAL_DECL_KERNEL(get_rows_q4_K);
    GGML_METAL_DECL_KERNEL(get_rows_q5_K);
    GGML_METAL_DECL_KERNEL(get_rows_q6_K);
    GGML_METAL_DECL_KERNEL(rms_norm);
    GGML_METAL_DECL_KERNEL(norm);
    GGML_METAL_DECL_KERNEL(mul_mat_f16_f32);
    GGML_METAL_DECL_KERNEL(mul_mat_q4_0_f32);
    GGML_METAL_DECL_KERNEL(mul_mat_q4_1_f32);
    GGML_METAL_DECL_KERNEL(mul_mat_q2_K_f32);
    GGML_METAL_DECL_KERNEL(mul_mat_q3_K_f32);
    GGML_METAL_DECL_KERNEL(mul_mat_q4_K_f32);
    GGML_METAL_DECL_KERNEL(mul_mat_q5_K_f32);
    GGML_METAL_DECL_KERNEL(mul_mat_q6_K_f32);
    GGML_METAL_DECL_KERNEL(rope);
    GGML_METAL_DECL_KERNEL(alibi_f32);
    GGML_METAL_DECL_KERNEL(cpy_f32_f16);
    GGML_METAL_DECL_KERNEL(cpy_f32_f32);
    GGML_METAL_DECL_KERNEL(cpy_f16_f16);

#undef GGML_METAL_DECL_KERNEL
};

// MSL code
// TODO: move the contents here when ready
//       for now it is easier to work in a separate file
static NSString * const msl_library_source = @"see metal.metal";

// Here to assist with NSBundle Path Hack
@interface GGMLMetalClass : NSObject
@end
@implementation GGMLMetalClass
@end

struct ggml_metal_context * ggml_metal_init(int n_cb) {
    fprintf(stderr, "%s: allocating\n", __func__);

    struct ggml_metal_context * ctx = malloc(sizeof(struct ggml_metal_context));

    ctx->n_cb   = n_cb;
    ctx->device = MTLCreateSystemDefaultDevice();
    ctx->queue  = [ctx->device newCommandQueue];
    ctx->n_buffers = 0;
    ctx->concur_list_len = 0;

    // determine if we can use MPS
    if (MPSSupportsMTLDevice(ctx->device)) {
        fprintf(stderr, "%s: using MPS\n", __func__);
    } else {
        fprintf(stderr, "%s: not using MPS\n", __func__);
        GGML_ASSERT(false && "MPS not supported");
    }

#if 0
    // compile from source string and show compile log
    {
        NSError * error = nil;

        ctx->library = [ctx->device newLibraryWithSource:msl_library_source options:nil error:&error];
        if (error) {
            fprintf(stderr, "%s: error: %s\n", __func__, [[error description] UTF8String]);
            exit(1);
        }
    }
#else
    UNUSED(msl_library_source);

    // read the source from "ggml-metal.metal" into a string and use newLibraryWithSource
    {
        NSError * error = nil;

        //NSString * path = [[NSBundle mainBundle] pathForResource:@"../../examples/metal/metal" ofType:@"metal"];
        // NSBundle * bundle = [NSBundle bundleForClass:[GGMLMetalClass class]];
        // NSString * path = [bundle pathForResource:@"ggml-metal" ofType:@"metal"];
        // fprintf(stderr, "%s: loading '%s'\n", __func__, [path UTF8String]);

        NSString * src  = @"#include <metal_stdlib>\n\nusing namespace metal;\n\n#define MAX(x, y) ((x) > (y) ? (x) : (y))\n\n#define QK4_0 32\n#define QR4_0 2\ntypedef struct {\n    half    d;             // delta\n    uint8_t qs[QK4_0 / 2]; // nibbles / quants\n} block_q4_0;\n\n#define QK4_1 32\ntypedef struct {\n    half d;          // delta\n    half m;          // min\n    uint8_t qs[QK4_1 / 2];  // nibbles / quants\n} block_q4_1;\n\nstatic void dequantize_row_q4_0(device const block_q4_0 * x, device float * y, int k) {\n    const int qk = QK4_0;\n\n    assert(k % qk == 0);\n\n    const int nb = k / qk;\n\n    for (int i = 0; i < nb; i++) {\n        const half d = x[i].d;\n\n        for (int j = 0; j < qk/2; ++j) {\n            const int x0 = (x[i].qs[j] & 0x0F) - 8;\n            const int x1 = (x[i].qs[j] >>   4) - 8;\n\n            y[i*qk + j + 0   ] = x0*d;\n            y[i*qk + j + qk/2] = x1*d;\n        }\n    }\n}\n\nstatic void dequantize_row_q4_1(device const block_q4_1 * x, device float * y, int k) {\n    const int qk = QK4_1;\n\n    assert(k % qk == 0);\n\n    const int nb = k / qk;\n\n    for (int i = 0; i < nb; i++) {\n        const half d = x[i].d;\n        const half m = x[i].m;\n\n        for (int j = 0; j < qk/2; ++j) {\n            const int x0 = (x[i].qs[j] & 0x0F);\n            const int x1 = (x[i].qs[j] >>   4);\n\n            y[i*qk + j + 0   ] = x0*d + m;\n            y[i*qk + j + qk/2] = x1*d + m;\n        }\n    }\n}\n\nkernel void kernel_add(\n        device const float * src0,\n        device const float * src1,\n        device       float * dst,\n        uint tpig[[thread_position_in_grid]]) {\n    dst[tpig] = src0[tpig] + src1[tpig];\n}\n\n// assumption: src1 is a row\n// broadcast src1 into src0\nkernel void kernel_add_row(\n        device const float * src0,\n        device const float * src1,\n        device       float * dst,\n        constant   int64_t & ne00,\n        uint tpig[[thread_position_in_grid]]) {\n    dst[tpig] = src0[tpig] + src1[tpig % ne00];\n}\n\nkernel void kernel_mul(\n        device const float * src0,\n        device const float * src1,\n        device       float * dst,\n        uint tpig[[thread_position_in_grid]]) {\n    dst[tpig] = src0[tpig] * src1[tpig];\n}\n\n// assumption: src1 is a row\n// broadcast src1 into src0\nkernel void kernel_mul_row(\n        device const float * src0,\n        device const float * src1,\n        device       float * dst,\n        constant   int64_t & ne00,\n        uint tpig[[thread_position_in_grid]]) {\n    dst[tpig] = src0[tpig] * src1[tpig % ne00];\n}\n\nkernel void kernel_scale(\n        device const float * src0,\n        device       float * dst,\n        constant     float & scale,\n        uint tpig[[thread_position_in_grid]]) {\n    dst[tpig] = src0[tpig] * scale;\n}\n\nkernel void kernel_silu(\n        device const float * src0,\n        device       float * dst,\n        uint tpig[[thread_position_in_grid]]) {\n    float x = src0[tpig];\n    dst[tpig] = x / (1.0f + exp(-x));\n}\n\nkernel void kernel_relu(\n        device const float * src0,\n        device       float * dst,\n        uint tpig[[thread_position_in_grid]]) {\n    dst[tpig] = max(0.0f, src0[tpig]);\n}\n\nconstant float GELU_COEF_A    = 0.044715f;\nconstant float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;\n\nkernel void kernel_gelu(\n    device const float * src0,\n    device       float * dst,\n    uint tpig[[thread_position_in_grid]]) {\n    float x = src0[tpig];\n    dst[tpig] = 0.5f*x*(1.0f + tanh(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)));\n}\n\nkernel void kernel_soft_max(\n        device const float * src0,\n        device       float * dst,\n        constant   int64_t & ne00,\n        constant   int64_t & ne01,\n        constant   int64_t & ne02,\n        threadgroup float  * buf [[threadgroup(0)]],\n        uint3 tgpig[[threadgroup_position_in_grid]],\n        uint3 tpitg[[thread_position_in_threadgroup]],\n        uint3   ntg[[threads_per_threadgroup]]) {\n    const int64_t i03 = tgpig[2];\n    const int64_t i02 = tgpig[1];\n    const int64_t i01 = tgpig[0];\n\n    device const float * psrc0 = src0 + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;\n    device       float * pdst  = dst  + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;\n\n    // parallel max\n    buf[tpitg[0]] = -INFINITY;\n    for (int i00 = tpitg[0]; i00 < ne00; i00 += ntg[0]) {\n        buf[tpitg[0]] = MAX(buf[tpitg[0]], psrc0[i00]);\n    }\n\n    // reduce\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n    for (uint i = ntg[0]/2; i > 0; i /= 2) {\n        if (tpitg[0] < i) {\n            buf[tpitg[0]] = MAX(buf[tpitg[0]], buf[tpitg[0] + i]);\n        }\n        threadgroup_barrier(mem_flags::mem_threadgroup);\n    }\n\n    // broadcast\n    if (tpitg[0] == 0) {\n        buf[0] = buf[0];\n    }\n\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n\n    const float max = buf[0];\n\n    // parallel sum\n    buf[tpitg[0]] = 0.0f;\n    for (int i00 = tpitg[0]; i00 < ne00; i00 += ntg[0]) {\n        buf[tpitg[0]] += exp(psrc0[i00] - max);\n    }\n\n    // reduce\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n    for (uint i = ntg[0]/2; i > 0; i /= 2) {\n        if (tpitg[0] < i) {\n            buf[tpitg[0]] += buf[tpitg[0] + i];\n        }\n        threadgroup_barrier(mem_flags::mem_threadgroup);\n    }\n\n    // broadcast\n    if (tpitg[0] == 0) {\n        buf[0] = buf[0];\n    }\n\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n\n    const float sum = buf[0];\n\n    for (int i00 = tpitg[0]; i00 < ne00; i00 += ntg[0]) {\n        pdst[i00] = exp(psrc0[i00] - max) / sum;\n    }\n}\n\nkernel void kernel_diag_mask_inf(\n        device const float * src0,\n        device       float * dst,\n        constant   int64_t & ne00,\n        constant   int64_t & ne01,\n        constant       int & n_past,\n        uint3 tpig[[thread_position_in_grid]]) {\n    const int64_t i02 = tpig[2];\n    const int64_t i01 = tpig[1];\n    const int64_t i00 = tpig[0];\n\n    if (i00 > n_past + i01) {\n        dst[i02*ne01*ne00 + i01*ne00 + i00] = -INFINITY;\n    } else {\n        dst[i02*ne01*ne00 + i01*ne00 + i00] = src0[i02*ne01*ne00 + i01*ne00 + i00];\n    }\n}\n\nkernel void kernel_get_rows_f16(\n        device const  void * src0,\n        device const   int * src1,\n        device       float * dst,\n        constant   int64_t & ne00,\n        constant  uint64_t & nb01,\n        constant  uint64_t & nb1,\n        uint tpig[[thread_position_in_grid]]) {\n    const int i = tpig;\n    const int r = ((device int32_t *) src1)[i];\n\n    for (int j = 0; j < ne00; j++) {\n        dst[i*nb1 + j] = ((device half *) ((device char *) src0 + r*nb01))[j];\n    }\n}\n\nkernel void kernel_get_rows_q4_0(\n        device const  void * src0,\n        device const   int * src1,\n        device       float * dst,\n        constant   int64_t & ne00,\n        constant  uint64_t & nb01,\n        constant  uint64_t & nb1,\n        uint tpig[[thread_position_in_grid]]) {\n    const int i = tpig;\n    const int r = ((device int32_t *) src1)[i];\n\n    dequantize_row_q4_0(\n            (device const block_q4_0 *) ((device char *) src0 + r*nb01),\n                       (device float *) ((device char *)  dst + i*nb1), ne00);\n}\n\nkernel void kernel_get_rows_q4_1(\n        device const  void * src0,\n        device const   int * src1,\n        device       float * dst,\n        constant   int64_t & ne00,\n        constant  uint64_t & nb01,\n        constant  uint64_t & nb1,\n        uint tpig[[thread_position_in_grid]]) {\n    const int i = tpig;\n    const int r = ((device int32_t *) src1)[i];\n\n    dequantize_row_q4_1(\n            (device const block_q4_1 *) ((device char *) src0 + r*nb01),\n                       (device float *) ((device char *)  dst + i*nb1), ne00);\n}\n\nkernel void kernel_norm(\n        device const  void * src0,\n        device       float * dst,\n        constant   int64_t & ne00,\n        constant  uint64_t & nb01,\n        constant     float & eps,\n        threadgroup float  * sum [[threadgroup(0)]],\n        uint tgpig[[threadgroup_position_in_grid]],\n        uint tpitg[[thread_position_in_threadgroup]],\n        uint   ntg[[threads_per_threadgroup]]) {\n    device const float * x = (device const float *) ((device const char *) src0 + tgpig*nb01);\n    // MEAN\n    // parallel sum\n    sum[tpitg] = 0.0f;\n    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {\n        sum[tpitg] += x[i00];\n    }\n    // reduce\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n    for (uint i = ntg/2; i > 0; i /= 2) {\n        if (tpitg < i) {\n            sum[tpitg] += sum[tpitg + i];\n        }\n        threadgroup_barrier(mem_flags::mem_threadgroup);\n    }\n    // broadcast\n    if (tpitg == 0) {\n        sum[0] /= ne00;\n    }\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n    const float mean  = sum[0];\n\n    // recenter\n    device float * y = dst + tgpig*ne00;\n    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {\n        y[i00] = x[i00] - mean;\n    }\n\n    // VARIANCE\n    // parallel sum\n    sum[tpitg] = 0.0f;\n    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {\n        sum[tpitg] += y[i00] * y[i00];\n    }\n    // reduce\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n    for (uint i = ntg/2; i > 0; i /= 2) {\n        if (tpitg < i) {\n            sum[tpitg] += sum[tpitg + i];\n        }\n        threadgroup_barrier(mem_flags::mem_threadgroup);\n    }\n    // broadcast\n    if (tpitg == 0) {\n        sum[0] /= ne00;\n    }\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n    const float variance = sum[0];\n\n    const float scale = 1.0f/sqrt(variance + eps);\n    for (int i00 = tpitg; i00 < ne00; i00 += ntg) {\n        y[i00] = y[i00] * scale;\n    }\n}\n\n\nkernel void kernel_rms_norm(\n        device const  void * src0,\n        device       float * dst,\n        constant   int64_t & ne00,\n        constant  uint64_t & nb01,\n        constant     float & eps,\n        threadgroup float  * sum [[threadgroup(0)]],\n        uint tgpig[[threadgroup_position_in_grid]],\n        uint tpitg[[thread_position_in_threadgroup]],\n        uint sgitg[[simdgroup_index_in_threadgroup]],\n        uint tiisg[[thread_index_in_simdgroup]],\n        uint   ntg[[threads_per_threadgroup]]) {\n    device const float4 * x = (device const float4 *) ((device const char *) src0 + tgpig*nb01);\n    device const float * x_scalar = (device const float *) x;\n    float4 sumf=0;\n    float all_sum=0;\n\n    // parallel sum\n    for (int i00 = tpitg; i00 < ne00/4; i00 += ntg) {\n        sumf += x[i00] * x[i00];\n    }\n    all_sum = sumf[0] + sumf[1] + sumf[2] + sumf[3];\n    all_sum = simd_sum(all_sum);\n    if (tiisg == 0) {\n        sum[sgitg] = all_sum;\n    }\n\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n    // broadcast, simd group number is ntg / 32\n    for (uint i = ntg / 32 / 2; i > 0; i /= 2) {\n       if (tpitg < i) {\n           sum[tpitg] += sum[tpitg + i];\n       }\n    }\n    if (tpitg == 0) {\n        for (int i = 4 * (ne00 / 4); i < ne00; i++) {sum[0] += x_scalar[i];}\n        sum[0] /= ne00;\n    }\n\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n\n    const float mean  = sum[0];\n    const float scale = 1.0f/sqrt(mean + eps);\n\n    device float4 * y = (device float4 *) (dst + tgpig*ne00);\n    device float * y_scalar = (device float *) y;\n    for (int i00 = tpitg; i00 < ne00/4; i00 += ntg) {\n        y[i00] = x[i00] * scale;\n    }\n    if (tpitg == 0) {\n        for (int i00 = 4 * (ne00 / 4); i00 < ne00; i00++) {y_scalar[i00] = x_scalar[i00] * scale;}\n    }\n}\n\n// function for calculate inner product between half a q4_0 block and 16 floats (yl), sumy is SUM(yl[i])\n// il indicates where the q4 quants begin (0 or QK4_0/4)\n// we assume that the yl's have been multiplied with the appropriate scale factor\n// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)\ninline float block_q_n_dot_y(device const block_q4_0 * qb_curr, float sumy, thread float * yl, int il) {\n    float d = qb_curr->d;\n    float2 acc = 0.f;\n    device const uint16_t * qs = ((device const uint16_t *)qb_curr + 1 + il/2);\n    for (int i = 0; i < 8; i+=2) {\n        acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F)\n                + yl[i + 1] * (qs[i / 2] & 0x0F00);\n        acc[1] += yl[i + 8] * (qs[i / 2] & 0x00F0)\n                + yl[i + 9] * (qs[i / 2] & 0xF000);\n    }\n    return d * (sumy * -8.f + acc[0] + acc[1]);\n}\n\n// function for calculate inner product between half a q4_1 block and 16 floats (yl), sumy is SUM(yl[i])\n// il indicates where the q4 quants begin (0 or QK4_0/4)\n// we assume that the yl's have been multiplied with the appropriate scale factor\n// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)\ninline float block_q_n_dot_y(device const block_q4_1 * qb_curr, float sumy, thread float * yl, int il) {\n    float d = qb_curr->d;\n    float m = qb_curr->m;\n    device const uint16_t * qs = ((device const uint16_t *)qb_curr + 2 + il/2);\n    float2 acc = 0.f;\n    for (int i = 0; i < 8; i+=2) {\n        acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F)\n                + yl[i + 1] * (qs[i / 2] & 0x0F00);\n        acc[1] += yl[i + 8] * (qs[i / 2] & 0x00F0)\n                + yl[i + 9] * (qs[i / 2] & 0xF000);\n    }\n    return d * (acc[0] + acc[1]) + sumy * m;\n}\n\n// putting them in the kernel cause a significant performance penalty\n#define N_DST 4 // each SIMD group works on 4 rows\n#define N_SIMDGROUP 2 // number of SIMD groups in a thread group\n#define N_SIMDWIDTH 32 // assuming SIMD group size is 32\n//Note: This is a template, but strictly speaking it only applies to\n//      quantizations where the block size is 32. It also does not\n//      giard against the number of rows not being divisible by\n//      N_DST, so this is another explicit assumption of the implementation.\ntemplate<typename block_q_type, int nr, int nsg, int nw>\nvoid mul_vec_q_n_f32(device const void * src0, device const float * src1, device float * dst,\n                    int64_t ne00, int64_t ne10, int64_t ne0, int64_t ne01,\n                    uint2 tgpig, uint tiisg, uint sgitg) {\n    const int nb = ne00/QK4_0;\n    const int r0 = tgpig.x;\n    const int r1 = tgpig.y;\n    const int first_row = (r0 * nsg + sgitg) * nr;\n    device const block_q_type * x = (device const block_q_type *) src0 + first_row * nb;\n    device const float      * y = (device const float      *) src1 + r1*ne10;\n    float yl[16];       // src1 vector cache\n    float sumf[nr]={0.f};\n\n    const int ix = tiisg/2;\n    const int il = 8*(tiisg%2);\n\n    device const float * yb = y + ix * QK4_0 + il;\n\n    // each thread in a SIMD group deals with half a block.\n    for (int ib = ix; ib < nb; ib += nw/2) {\n        float sumy = 0;\n        for (int i = 0; i < 8; i += 2) {\n            sumy += yb[i] + yb[i+1];\n            yl[i+0] = yb[i+ 0];\n            yl[i+1] = yb[i+ 1]/256.f;\n            sumy += yb[i+16] + yb[i+17];\n            yl[i+8] = yb[i+16]/16.f;\n            yl[i+9] = yb[i+17]/4096.f;\n        }\n\n        for (int row = 0; row < nr; row++) {\n            sumf[row] += block_q_n_dot_y(x+ib+row*nb, sumy, yl, il);\n        }\n\n        yb += QK4_0 * 16;\n    }\n\n    for (int row = 0; row < nr; ++row) {\n        const float tot = simd_sum(sumf[row]);\n        if (tiisg == 0 && first_row + row < ne01) {\n            dst[r1*ne0 + first_row + row] = tot;\n        }\n    }\n}\n\nkernel void kernel_mul_mat_q4_0_f32(\n        device const  void * src0,\n        device const float * src1,\n        device       float * dst,\n        constant   int64_t & ne00,\n        constant   int64_t & ne10,\n        constant   int64_t & ne0,\n        constant   int64_t & ne01[[buffer(4)]],\n        uint2 tgpig[[threadgroup_position_in_grid]],\n        uint tiisg[[thread_index_in_simdgroup]],\n        uint sgitg[[simdgroup_index_in_threadgroup]]) {\n    mul_vec_q_n_f32<block_q4_0, N_DST, N_SIMDGROUP, N_SIMDWIDTH>(src0,src1,dst,ne00,ne10,ne0,ne01,tgpig,tiisg,sgitg);\n}\n\nkernel void kernel_mul_mat_q4_1_f32(\n        device const  void * src0,\n        device const float * src1,\n        device       float * dst,\n        constant   int64_t & ne00,\n        constant   int64_t & ne10,\n        constant   int64_t & ne0,\n        constant   int64_t & ne01[[buffer(4)]],\n        uint2 tgpig[[threadgroup_position_in_grid]],\n        uint tiisg[[thread_index_in_simdgroup]],\n        uint sgitg[[simdgroup_index_in_threadgroup]]) {\n     mul_vec_q_n_f32<block_q4_1, N_DST, N_SIMDGROUP, N_SIMDWIDTH>(src0,src1,dst,ne00,ne10,ne0,ne01,tgpig,tiisg,sgitg);\n}\n\nkernel void kernel_mul_mat_f16_f32(\n        device const  char * src0,\n        device const  char * src1,\n        device       float * dst,\n        constant   int64_t & ne00,\n        constant   int64_t & ne01,\n        constant   int64_t & ne02,\n        constant  uint64_t & nb00,\n        constant  uint64_t & nb01,\n        constant  uint64_t & nb02,\n        constant   int64_t & ne10,\n        constant   int64_t & ne11,\n        constant   int64_t & ne12,\n        constant  uint64_t & nb10,\n        constant  uint64_t & nb11,\n        constant  uint64_t & nb12,\n        constant   int64_t & ne0,\n        constant   int64_t & ne1,\n        threadgroup float  * sum [[threadgroup(0)]],\n        uint3 tgpig[[threadgroup_position_in_grid]],\n        uint3  tpig[[thread_position_in_grid]],\n        uint3 tpitg[[thread_position_in_threadgroup]],\n        uint3  tptg[[threads_per_threadgroup]]) {\n\n    const int64_t r0 = tgpig.x;\n    const int64_t r1 = tgpig.y;\n    const int64_t im = tgpig.z;\n\n    device const half  * x = (device const half  *) (src0 + r0*nb01 + im/(ne12/ne02)*nb02);\n    device const float * y = (device const float *) (src1 + r1*nb11 + im*nb12);\n\n    sum[tpitg.x] = 0.0f;\n\n    for (int i = tpitg.x; i < ne00; i += tptg.x) {\n        sum[tpitg.x] += (float) x[i] * (float) y[i];\n    }\n\n    // accumulate the sum from all threads in the threadgroup\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n    for (uint i = tptg.x/2; i > 0; i /= 2) {\n        if (tpitg.x < i) {\n            sum[tpitg.x] += sum[tpitg.x + i];\n        }\n        threadgroup_barrier(mem_flags::mem_threadgroup);\n    }\n\n    if (tpitg.x == 0) {\n        dst[im*ne1*ne0 + r1*ne0 + r0] = sum[0];\n    }\n}\n\n\nkernel void kernel_alibi_f32(\n        device const float * src0,\n        device       float * dst,\n        constant   int64_t & ne00,\n        constant   int64_t & ne01,\n        constant   int64_t & ne02,\n        constant   int64_t & ne03,\n        constant  uint64_t & nb00,\n        constant  uint64_t & nb01,\n        constant  uint64_t & nb02,\n        constant  uint64_t & nb03,\n        constant   int64_t & ne0,\n        constant   int64_t & ne1,\n        constant   int64_t & ne2,\n        constant   int64_t & ne3,\n        constant  uint64_t & nb0,\n        constant  uint64_t & nb1,\n        constant  uint64_t & nb2,\n        constant  uint64_t & nb3,\n        constant      float & m0,\n        uint3 tgpig[[threadgroup_position_in_grid]],\n        uint3 tpitg[[thread_position_in_threadgroup]],\n        uint3   ntg[[threads_per_threadgroup]]) {\n    const int64_t i03 = tgpig[2];\n    const int64_t i02 = tgpig[1];\n    const int64_t i01 = tgpig[0];\n\n    const int64_t n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;\n\n    const int64_t i3 = n / (ne2*ne1*ne0);\n    const int64_t i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);\n    const int64_t i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;\n    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);\n\n    device float * dst_data = (device float *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);\n    float m_k = pow(m0, i2 + 1);\n    for (int64_t i00 = tpitg.x; i00 < ne00; i00 += ntg.x) {\n        device const float * src = (device float *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);\n        dst_data[i00] = src[0] + m_k * (i00 - ne00 + 1);\n    }\n}\n\nkernel void kernel_rope(\n        device const  void * src0,\n        device       float * dst,\n        constant   int64_t & ne00,\n        constant   int64_t & ne01,\n        constant   int64_t & ne02,\n        constant   int64_t & ne03,\n        constant  uint64_t & nb00,\n        constant  uint64_t & nb01,\n        constant  uint64_t & nb02,\n        constant  uint64_t & nb03,\n        constant   int64_t & ne0,\n        constant   int64_t & ne1,\n        constant   int64_t & ne2,\n        constant   int64_t & ne3,\n        constant  uint64_t & nb0,\n        constant  uint64_t & nb1,\n        constant  uint64_t & nb2,\n        constant  uint64_t & nb3,\n        constant       int & n_past,\n        constant       int & n_dims,\n        constant       int & mode,\n        constant     float & freq_base,\n        constant     float & freq_scale,\n        uint3 tpig[[thread_position_in_grid]]) {\n    const int64_t i3 = tpig[2];\n    const int64_t i2 = tpig[1];\n    const int64_t i1 = tpig[0];\n\n    const bool is_neox = mode & 2;\n    const float theta_scale = pow(freq_base, -2.0f/n_dims);\n\n    const int64_t p = ((mode & 1) == 0 ? n_past + i2 : i2);\n\n    float theta = freq_scale * (float)p;\n\n    if (!is_neox) {\n        for (int64_t i0 = 0; i0 < ne0; i0 += 2) {\n            const float cos_theta = cos(theta);\n            const float sin_theta = sin(theta);\n\n            theta *= theta_scale;\n\n            device const float * const src = (device float *)((device char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);\n            device       float * dst_data  = (device float *)((device char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);\n\n            const float x0 = src[0];\n            const float x1 = src[1];\n\n            dst_data[0] = x0*cos_theta - x1*sin_theta;\n            dst_data[1] = x0*sin_theta + x1*cos_theta;\n        }\n    } else {\n        // TODO: implement\n    }\n}\n\nkernel void kernel_cpy_f16_f16(\n        device const half * src0,\n        device       half * dst,\n        constant   int64_t & ne00,\n        constant   int64_t & ne01,\n        constant   int64_t & ne02,\n        constant   int64_t & ne03,\n        constant  uint64_t & nb00,\n        constant  uint64_t & nb01,\n        constant  uint64_t & nb02,\n        constant  uint64_t & nb03,\n        constant   int64_t & ne0,\n        constant   int64_t & ne1,\n        constant   int64_t & ne2,\n        constant   int64_t & ne3,\n        constant  uint64_t & nb0,\n        constant  uint64_t & nb1,\n        constant  uint64_t & nb2,\n        constant  uint64_t & nb3,\n        uint3 tgpig[[threadgroup_position_in_grid]],\n        uint3 tpitg[[thread_position_in_threadgroup]],\n        uint3   ntg[[threads_per_threadgroup]]) {\n    const int64_t i03 = tgpig[2];\n    const int64_t i02 = tgpig[1];\n    const int64_t i01 = tgpig[0];\n\n    const int64_t n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;\n\n    const int64_t i3 = n / (ne2*ne1*ne0);\n    const int64_t i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);\n    const int64_t i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;\n    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);\n\n    device half * dst_data = (device half *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);\n\n    for (int64_t i00 = tpitg.x; i00 < ne00; i00 += ntg.x) {\n        device const half * src = (device half *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);\n        dst_data[i00] = src[0];\n    }\n}\n\nkernel void kernel_cpy_f32_f16(\n        device const float * src0,\n        device        half * dst,\n        constant   int64_t & ne00,\n        constant   int64_t & ne01,\n        constant   int64_t & ne02,\n        constant   int64_t & ne03,\n        constant  uint64_t & nb00,\n        constant  uint64_t & nb01,\n        constant  uint64_t & nb02,\n        constant  uint64_t & nb03,\n        constant   int64_t & ne0,\n        constant   int64_t & ne1,\n        constant   int64_t & ne2,\n        constant   int64_t & ne3,\n        constant  uint64_t & nb0,\n        constant  uint64_t & nb1,\n        constant  uint64_t & nb2,\n        constant  uint64_t & nb3,\n        uint3 tgpig[[threadgroup_position_in_grid]],\n        uint3 tpitg[[thread_position_in_threadgroup]],\n        uint3   ntg[[threads_per_threadgroup]]) {\n    const int64_t i03 = tgpig[2];\n    const int64_t i02 = tgpig[1];\n    const int64_t i01 = tgpig[0];\n\n    const int64_t n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;\n\n    const int64_t i3 = n / (ne2*ne1*ne0);\n    const int64_t i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);\n    const int64_t i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;\n    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);\n\n    device half * dst_data = (device half *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);\n\n    for (int64_t i00 = tpitg.x; i00 < ne00; i00 += ntg.x) {\n        device const float * src = (device float *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);\n\n        dst_data[i00] = src[0];\n    }\n}\n\nkernel void kernel_cpy_f32_f32(\n        device const float * src0,\n        device       float * dst,\n        constant   int64_t & ne00,\n        constant   int64_t & ne01,\n        constant   int64_t & ne02,\n        constant   int64_t & ne03,\n        constant  uint64_t & nb00,\n        constant  uint64_t & nb01,\n        constant  uint64_t & nb02,\n        constant  uint64_t & nb03,\n        constant   int64_t & ne0,\n        constant   int64_t & ne1,\n        constant   int64_t & ne2,\n        constant   int64_t & ne3,\n        constant  uint64_t & nb0,\n        constant  uint64_t & nb1,\n        constant  uint64_t & nb2,\n        constant  uint64_t & nb3,\n        uint3 tgpig[[threadgroup_position_in_grid]],\n        uint3 tpitg[[thread_position_in_threadgroup]],\n        uint3   ntg[[threads_per_threadgroup]]) {\n    const int64_t i03 = tgpig[2];\n    const int64_t i02 = tgpig[1];\n    const int64_t i01 = tgpig[0];\n\n    const int64_t n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;\n\n    const int64_t i3 = n / (ne2*ne1*ne0);\n    const int64_t i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);\n    const int64_t i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;\n    const int64_t i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);\n\n    device float * dst_data = (device float *) ((device char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);\n\n    for (int64_t i00 = tpitg.x; i00 < ne00; i00 += ntg.x) {\n        device const float * src = (device float *)((device char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);\n\n        dst_data[i00] = src[0];\n    }\n}\n\n//============================================ k-quants ======================================================\n\n#ifndef QK_K\n#define QK_K 256\n#else\nstatic_assert(QK_K == 256 || QK_K == 64, \"QK_K must be 256 or 64\");\n#endif\n\n#if QK_K == 256\n#define K_SCALE_SIZE 12\n#else\n#define K_SCALE_SIZE 4\n#endif\n\ntypedef struct {\n    uint8_t scales[QK_K/16]; // scales and mins, quantized with 4 bits\n    uint8_t qs[QK_K/4];      // quants\n    half d;           // super-block scale for quantized scales\n    half dmin;        // super-block scale for quantized mins\n} block_q2_K;\n// 84 bytes / block\n\ntypedef struct {\n    uint8_t hmask[QK_K/8];     // quants - high bit\n    uint8_t qs[QK_K/4];        // quants - low 2 bits\n#if QK_K == 64\n    uint8_t scales[2];\n#else\n    uint8_t scales[K_SCALE_SIZE]; // scales, quantized with 6 bits\n#endif\n    half d;             // super-block scale\n} block_q3_K;\n\n#if QK_K == 64\ntypedef struct {\n    half    d[2];          // super-block scales/mins\n    uint8_t scales[2];\n    uint8_t qs[QK_K/2];    // 4-bit quants\n} block_q4_K;\n#else\ntypedef struct {\n    half d;             // super-block scale for quantized scales\n    half dmin;          // super-block scale for quantized mins\n    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits\n    uint8_t qs[QK_K/2];        // 4--bit quants\n} block_q4_K;\n#endif\n\n#if QK_K == 64\ntypedef struct {\n    half  d;                     // super-block scales/mins\n    int8_t  scales[QK_K/16];     // 8-bit block scales\n    uint8_t qh[QK_K/8];          // quants, high bit\n    uint8_t qs[QK_K/2];          // quants, low 4 bits\n} block_q5_K;\n#else\ntypedef struct {\n    half d;                      // super-block scale for quantized scales\n    half dmin;                   // super-block scale for quantized mins\n    uint8_t scales[3*QK_K/64];   // scales and mins, quantized with 6 bits\n    uint8_t qh[QK_K/8];          // quants, high bit\n    uint8_t qs[QK_K/2];          // quants, low 4 bits\n} block_q5_K;\n// 176 bytes / block\n#endif\n\ntypedef struct {\n    uint8_t ql[QK_K/2];      // quants, lower 4 bits\n    uint8_t qh[QK_K/4];      // quants, upper 2 bits\n    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits\n    half d;                  // super-block scale\n} block_q6_K;\n// 210 bytes / block\n\nstatic inline uchar4 get_scale_min_k4(int j, device const uint8_t * q) {\n    uchar4 r;\n    if (j < 4) {\n        r[0] = q[j+0] & 63;\n        r[2] = q[j+1] & 63;\n        r[1] = q[j+4] & 63;\n        r[3] = q[j+5] & 63;\n    } else {\n        r[0] = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);\n        r[2] = (q[j+5] & 0xF) | ((q[j-3] >> 6) << 4);\n        r[1] = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);\n        r[3] = (q[j+5] >>  4) | ((q[j+1] >> 6) << 4);\n    }\n    return r;\n}\n\n//========================================== dequantization =============================\n\nstatic void dequantize_row_q2_K(device const block_q2_K * x, device float * y, int k) {\n    assert(k % QK_K == 0);\n    const int nb = k / QK_K;\n\n    for (int i = 0; i < nb; i++) {\n\n        const float d = x[i].d;\n        const float min = x[i].dmin;\n\n        device const uint8_t * q = x[i].qs;\n\n#if QK_K == 256\n        int is = 0;\n        float dl, ml;\n        for (int n = 0; n < QK_K; n += 128) {\n            int shift = 0;\n            for (int j = 0; j < 4; ++j) {\n\n                uint8_t sc = x[i].scales[is++];\n                dl = d * (sc & 0xF); ml = min * (sc >> 4);\n                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;\n\n                sc = x[i].scales[is++];\n                dl = d * (sc & 0xF); ml = min * (sc >> 4);\n                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l+16] >> shift) & 3)) - ml;\n\n                shift += 2;\n            }\n            q += 32;\n        }\n#else\n        float dl1 = d * (x[i].scales[0] & 0xF), ml1 = min * (x[i].scales[0] >> 4);\n        float dl2 = d * (x[i].scales[1] & 0xF), ml2 = min * (x[i].scales[1] >> 4);\n        float dl3 = d * (x[i].scales[2] & 0xF), ml3 = min * (x[i].scales[2] >> 4);\n        float dl4 = d * (x[i].scales[3] & 0xF), ml4 = min * (x[i].scales[3] >> 4);\n        for (int l = 0; l < 16; ++l) {\n            y[l+ 0] = dl1 * ((q[l] >> 0) & 3) - ml1;\n            y[l+16] = dl2 * ((q[l] >> 2) & 3) - ml2;\n            y[l+32] = dl3 * ((q[l] >> 4) & 3) - ml3;\n            y[l+48] = dl4 * ((q[l] >> 6) & 3) - ml4;\n        }\n        y += QK_K;\n#endif\n\n    }\n}\n\nstatic void dequantize_row_q3_K(device const block_q3_K * x, device float * y, int k) {\n    assert(k % QK_K == 0);\n    const int nb = k / QK_K;\n\n#if QK_K == 256\n\n    const uint16_t kmask1 = 0x0303;\n    const uint16_t kmask2 = 0x0f0f;\n\n    uint16_t aux[8];\n    thread const int8_t * scales = (thread const int8_t*)aux;\n\n    for (int i = 0; i < nb; i++) {\n\n        const float d_all = (float)(x[i].d);\n\n        device const uint8_t * q = x[i].qs;\n        device const uint8_t * h = x[i].hmask;\n        uint8_t m = 1;\n\n        device const uint16_t * a = (device const uint16_t *)x[i].scales;\n        aux[0] = (a[0] & kmask2) | (((a[4] >> 0) & kmask1) << 4);\n        aux[1] = (a[1] & kmask2) | (((a[5] >> 0) & kmask1) << 4);\n        aux[2] = (a[2] & kmask2) | (((a[4] >> 2) & kmask1) << 4);\n        aux[3] = (a[3] & kmask2) | (((a[5] >> 2) & kmask1) << 4);\n        aux[4] = ((a[0] >> 4) & kmask2) | (((a[4] >> 4) & kmask1) << 4);\n        aux[5] = ((a[1] >> 4) & kmask2) | (((a[5] >> 4) & kmask1) << 4);\n        aux[6] = ((a[2] >> 4) & kmask2) | (((a[4] >> 6) & kmask1) << 4);\n        aux[7] = ((a[3] >> 4) & kmask2) | (((a[5] >> 6) & kmask1) << 4);\n\n        int is = 0;\n        float dl;\n        for (int n = 0; n < QK_K; n += 128) {\n            int shift = 0;\n            for (int j = 0; j < 4; ++j) {\n\n                dl = d_all * (scales[is++] - 32);\n                for (int l = 0; l < 16; ++l) {\n                    *y++ = dl * ((int8_t)((q[l+ 0] >> shift) & 3) - ((h[l+ 0] & m) ? 0 : 4));\n                }\n\n                dl = d_all * (scales[is++] - 32);\n                for (int l = 0; l < 16; ++l) {\n                    *y++ = dl * ((int8_t)((q[l+16] >> shift) & 3) - ((h[l+16] & m) ? 0 : 4));\n                }\n\n                shift += 2;\n                m <<= 1;\n            }\n            q += 32;\n        }\n    }\n#else\n    for (int i = 0; i < nb; i++) {\n\n        const float d_all = (float)(x[i].d);\n\n        device const uint8_t * q = x[i].qs;\n        device const uint8_t * hm = x[i].hmask;\n\n        const float d1 = d_all * ((x[i].scales[0] & 0xF) - 8);\n        const float d2 = d_all * ((x[i].scales[0] >>  4) - 8);\n        const float d3 = d_all * ((x[i].scales[1] & 0xF) - 8);\n        const float d4 = d_all * ((x[i].scales[1] >>  4) - 8);\n\n        for (int l = 0; l < 8; ++l) {\n            uint8_t h = hm[l];\n            y[l+ 0] = d1 * ((int8_t)((q[l+0] >> 0) & 3) - ((h & 0x01) ? 0 : 4));\n            y[l+ 8] = d1 * ((int8_t)((q[l+8] >> 0) & 3) - ((h & 0x02) ? 0 : 4));\n            y[l+16] = d2 * ((int8_t)((q[l+0] >> 2) & 3) - ((h & 0x04) ? 0 : 4));\n            y[l+24] = d2 * ((int8_t)((q[l+8] >> 2) & 3) - ((h & 0x08) ? 0 : 4));\n            y[l+32] = d3 * ((int8_t)((q[l+0] >> 4) & 3) - ((h & 0x10) ? 0 : 4));\n            y[l+40] = d3 * ((int8_t)((q[l+8] >> 4) & 3) - ((h & 0x20) ? 0 : 4));\n            y[l+48] = d4 * ((int8_t)((q[l+0] >> 6) & 3) - ((h & 0x40) ? 0 : 4));\n            y[l+56] = d4 * ((int8_t)((q[l+8] >> 6) & 3) - ((h & 0x80) ? 0 : 4));\n        }\n        y += QK_K;\n    }\n#endif\n\n}\n\nstatic void dequantize_row_q4_K(device const block_q4_K * x, device float * y, int k) {\n    assert(k % QK_K == 0);\n    const int nb = k / QK_K;\n\n    for (int i = 0; i < nb; i++) {\n\n        device const uint8_t * q = x[i].qs;\n\n#if QK_K == 256\n        const float d = x[i].d;\n        const float min = x[i].dmin;\n\n        device const uint8_t * scales = x[i].scales;\n\n        int is = 0;\n        for (int j = 0; j < QK_K; j += 64) {\n            const uchar4 sc = get_scale_min_k4(is, scales);\n            const float d1 = d * sc[0]; const float m1 = min * sc[1];\n            const float d2 = d * sc[2]; const float m2 = min * sc[3];\n            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;\n            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l]  >> 4) - m2;\n            q += 32; is += 2;\n        }\n#else\n        device const uint8_t * s = x[i].scales;\n        device const half2 * dh = (device const half2 *)x[i].d;\n        const float2 d = (float2)dh[0];\n        const float d1 = d[0] * (s[0] & 0xF);\n        const float d2 = d[0] * (s[1] & 0xF);\n        const float m1 = d[1] * (s[0] >>  4);\n        const float m2 = d[1] * (s[1] >>  4);\n        for (int l = 0; l < 32; ++l) {\n            y[l+ 0] = d1 * (q[l] & 0xF) - m1;\n            y[l+32] = d2 * (q[l] >>  4) - m2;\n        }\n        y += QK_K;\n#endif\n\n    }\n}\n\nstatic void dequantize_row_q5_K(device const block_q5_K * x, device float * y, int k) {\n    assert(k % QK_K == 0);\n    const int nb = k / QK_K;\n\n#if QK_K == 256\n   for (int i = 0; i < nb; i++) {\n\n        const float d = (float)(x[i].d);\n        const float min = (float)(x[i].dmin);\n\n        device const uint8_t * ql = x[i].qs;\n        device const uint8_t * qh = x[i].qh;\n\n        int is = 0;\n        uint8_t u1 = 1, u2 = 2;\n        for (int j = 0; j < QK_K; j += 64) {\n            const uchar4 sc = get_scale_min_k4(is, x[i].scales);\n            const float d1 = d * sc[0]; const float m1 = min * sc[1];\n            const float d2 = d * sc[2]; const float m2 = min * sc[3];\n            for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;\n            for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l]  >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;\n            ql += 32; is += 2;\n            u1 <<= 2; u2 <<= 2;\n        }\n    }\n#else\n    for (int i = 0; i < nb; i++) {\n\n        const float d = (float)x[i].d;\n\n        device const uint8_t * ql = x[i].qs;\n        device const uint8_t * qh = x[i].qh;\n        device const int8_t  * sc = x[i].scales;\n\n        for (int l = 0; l < 8; ++l) {\n            y[l+ 0] = d * sc[0] * ((ql[l+ 0] & 0xF) - (qh[l] & 0x01 ? 0 : 16));\n            y[l+ 8] = d * sc[0] * ((ql[l+ 8] & 0xF) - (qh[l] & 0x02 ? 0 : 16));\n            y[l+16] = d * sc[1] * ((ql[l+16] & 0xF) - (qh[l] & 0x04 ? 0 : 16));\n            y[l+24] = d * sc[1] * ((ql[l+24] & 0xF) - (qh[l] & 0x08 ? 0 : 16));\n            y[l+32] = d * sc[2] * ((ql[l+ 0] >>  4) - (qh[l] & 0x10 ? 0 : 16));\n            y[l+40] = d * sc[2] * ((ql[l+ 8] >>  4) - (qh[l] & 0x20 ? 0 : 16));\n            y[l+48] = d * sc[3] * ((ql[l+16] >>  4) - (qh[l] & 0x40 ? 0 : 16));\n            y[l+56] = d * sc[3] * ((ql[l+24] >>  4) - (qh[l] & 0x80 ? 0 : 16));\n        }\n        y += QK_K;\n    }\n#endif\n\n}\n\nstatic void dequantize_row_q6_K(device const block_q6_K * x, device float * y, int k) {\n    assert(k % QK_K == 0);\n    const int nb = k / QK_K;\n\n    for (int i = 0; i < nb; i++) {\n\n        device const uint8_t * ql = x[i].ql;\n        device const uint8_t * qh = x[i].qh;\n        device const int8_t  * sc = x[i].scales;\n\n        const float d = x[i].d;\n\n#if QK_K == 256\n        for (int n = 0; n < QK_K; n += 128) {\n            for (int l = 0; l < 32; ++l) {\n                int is = l/16;\n                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;\n                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;\n                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;\n                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;\n                y[l +  0] = d * sc[is + 0] * q1;\n                y[l + 32] = d * sc[is + 2] * q2;\n                y[l + 64] = d * sc[is + 4] * q3;\n                y[l + 96] = d * sc[is + 6] * q4;\n            }\n            y  += 128;\n            ql += 64;\n            qh += 32;\n            sc += 8;\n        }\n#else\n        for (int l = 0; l < 16; ++l) {\n            const int8_t q1 = (int8_t)((ql[l+ 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;\n            const int8_t q2 = (int8_t)((ql[l+16] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;\n            const int8_t q3 = (int8_t)((ql[l+ 0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;\n            const int8_t q4 = (int8_t)((ql[l+16]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;\n            y[l+ 0] = d * sc[0] * q1;\n            y[l+16] = d * sc[1] * q2;\n            y[l+32] = d * sc[2] * q3;\n            y[l+48] = d * sc[3] * q4;\n        }\n        y  += 64;\n#endif\n    }\n}\n\nkernel void kernel_get_rows_q2_K(\n        device const  void * src0,\n        device const   int * src1,\n        device       float * dst,\n        constant   int64_t & ne00,\n        constant  uint64_t & nb01,\n        constant  uint64_t & nb1,\n        uint tpig[[thread_position_in_grid]]) {\n    const int i = tpig;\n    const int r = ((device int32_t *) src1)[i];\n\n    dequantize_row_q2_K(\n            (device const block_q2_K *) ((device char *) src0 + r*nb01),\n                       (device float *) ((device char *)  dst + i*nb1), ne00);\n}\n\nkernel void kernel_get_rows_q3_K(\n        device const  void * src0,\n        device const   int * src1,\n        device       float * dst,\n        constant   int64_t & ne00,\n        constant  uint64_t & nb01,\n        constant  uint64_t & nb1,\n        uint tpig[[thread_position_in_grid]]) {\n    const int i = tpig;\n    const int r = ((device int32_t *) src1)[i];\n\n    dequantize_row_q3_K(\n            (device const block_q3_K *) ((device char *) src0 + r*nb01),\n                       (device float *) ((device char *)  dst + i*nb1), ne00);\n}\n\nkernel void kernel_get_rows_q4_K(\n        device const  void * src0,\n        device const   int * src1,\n        device       float * dst,\n        constant   int64_t & ne00,\n        constant  uint64_t & nb01,\n        constant  uint64_t & nb1,\n        uint tpig[[thread_position_in_grid]]) {\n    const int i = tpig;\n    const int r = ((device int32_t *) src1)[i];\n\n    dequantize_row_q4_K(\n            (device const block_q4_K *) ((device char *) src0 + r*nb01),\n                       (device float *) ((device char *)  dst + i*nb1), ne00);\n}\n\nkernel void kernel_get_rows_q5_K(\n        device const  void * src0,\n        device const   int * src1,\n        device       float * dst,\n        constant   int64_t & ne00,\n        constant  uint64_t & nb01,\n        constant  uint64_t & nb1,\n        uint tpig[[thread_position_in_grid]]) {\n    const int i = tpig;\n    const int r = ((device int32_t *) src1)[i];\n\n    dequantize_row_q5_K(\n            (device const block_q5_K *) ((device char *) src0 + r*nb01),\n                       (device float *) ((device char *)  dst + i*nb1), ne00);\n}\n\nkernel void kernel_get_rows_q6_K(\n        device const  void * src0,\n        device const   int * src1,\n        device       float * dst,\n        constant   int64_t & ne00,\n        constant  uint64_t & nb01,\n        constant  uint64_t & nb1,\n        uint tpig[[thread_position_in_grid]]) {\n    const int i = tpig;\n    const int r = ((device int32_t *) src1)[i];\n\n    dequantize_row_q6_K(\n            (device const block_q6_K *) ((device char *) src0 + r*nb01),\n                       (device float *) ((device char *)  dst + i*nb1), ne00);\n}\n\n//====================================== dot products =========================\n\nkernel void kernel_mul_mat_q2_K_f32(\n        device const  void * src0,\n        device const float * src1,\n        device       float * dst,\n        constant   int64_t & ne00,\n        constant   int64_t & ne10,\n        constant   int64_t & ne0,\n        constant   int64_t & ne01[[buffer(4)]],\n        uint2 tgpig[[threadgroup_position_in_grid]],\n        uint tiisg[[thread_index_in_simdgroup]],\n        uint sgitg[[simdgroup_index_in_threadgroup]]) {\n\n    const int nb = ne00/QK_K;\n    const int r0 = tgpig.x;\n    const int r1 = tgpig.y;\n\n    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;\n    const int ib_row = first_row * nb;\n    device const block_q2_K * x = (device const block_q2_K *) src0 + ib_row;\n    device const float      * y = (device const float      *) src1 + r1*ne10;\n    float yl[32];\n    float sumf[N_DST]={0.f}, all_sum;\n\n    const int step = sizeof(block_q2_K) * nb;\n\n#if QK_K == 256\n    const int ix = tiisg/8;  // 0...3\n    const int it = tiisg%8;  // 0...7\n    const int im = it/4;     // 0 or 1\n    const int ir = it%4;     // 0...3\n    const int is = (8*ir)/16;// 0 or 1\n\n    device const float * y4 = y + ix * QK_K + 128 * im + 8 * ir;\n\n    for (int ib = ix; ib < nb; ib += 4) {\n\n        float4 sumy = {0.f, 0.f, 0.f, 0.f};\n        for (int i = 0; i < 8; ++i) {\n            yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];\n            yl[i+ 8] = y4[i+32]; sumy[1] += yl[i+ 8];\n            yl[i+16] = y4[i+64]; sumy[2] += yl[i+16];\n            yl[i+24] = y4[i+96]; sumy[3] += yl[i+24];\n        }\n\n        device const uint8_t  * sc = (device const uint8_t  *)x[ib].scales + 8*im + is;\n        device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 16 * im + 4 * ir;\n        device const half     * dh = &x[ib].d;\n\n        for (int row = 0; row < N_DST; row++) {\n\n            float4 acc1 = {0.f, 0.f, 0.f, 0.f};\n            float4 acc2 = {0.f, 0.f, 0.f, 0.f};\n            for (int i = 0; i < 8; i += 2) {\n                acc1[0] += yl[i+ 0] * (qs[i/2] & 0x0003);\n                acc2[0] += yl[i+ 1] * (qs[i/2] & 0x0300);\n                acc1[1] += yl[i+ 8] * (qs[i/2] & 0x000c);\n                acc2[1] += yl[i+ 9] * (qs[i/2] & 0x0c00);\n                acc1[2] += yl[i+16] * (qs[i/2] & 0x0030);\n                acc2[2] += yl[i+17] * (qs[i/2] & 0x3000);\n                acc1[3] += yl[i+24] * (qs[i/2] & 0x00c0);\n                acc2[3] += yl[i+25] * (qs[i/2] & 0xc000);\n            }\n            float dall = dh[0];\n            float dmin = dh[1] * 1.f/16.f;\n            sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * 1.f/ 1.f +\n                                 (acc1[1] + 1.f/256.f * acc2[1]) * (sc[2] & 0xF) * 1.f/ 4.f +\n                                 (acc1[2] + 1.f/256.f * acc2[2]) * (sc[4] & 0xF) * 1.f/16.f +\n                                 (acc1[3] + 1.f/256.f * acc2[3]) * (sc[6] & 0xF) * 1.f/64.f) -\n                         dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) + sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));\n\n            qs += step/2;\n            sc += step;\n            dh += step/2;\n        }\n\n        y4 += 4 * QK_K;\n    }\n#else\n    const int ix = tiisg/2;  // 0...15\n    const int it = tiisg%2;  // 0...1\n\n    device const float * y4 = y + ix * QK_K + 8 * it;\n\n    for (int ib = ix; ib < nb; ib += 16) {\n\n        float4 sumy = {0.f, 0.f, 0.f, 0.f};\n        for (int i = 0; i < 8; ++i) {\n            yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];\n            yl[i+ 8] = y4[i+16]; sumy[1] += yl[i+ 8];\n            yl[i+16] = y4[i+32]; sumy[2] += yl[i+16];\n            yl[i+24] = y4[i+48]; sumy[3] += yl[i+24];\n        }\n\n        device const uint8_t  * sc = (device const uint8_t  *)x[ib].scales;\n        device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 4 * it;\n        device const half     * dh = &x[ib].d;\n\n        for (int row = 0; row < N_DST; row++) {\n\n            float4 acc1 = {0.f, 0.f, 0.f, 0.f};\n            float4 acc2 = {0.f, 0.f, 0.f, 0.f};\n            for (int i = 0; i < 8; i += 2) {\n                acc1[0] += yl[i+ 0] * (qs[i/2] & 0x0003);\n                acc2[0] += yl[i+ 1] * (qs[i/2] & 0x0300);\n                acc1[1] += yl[i+ 8] * (qs[i/2] & 0x000c);\n                acc2[1] += yl[i+ 9] * (qs[i/2] & 0x0c00);\n                acc1[2] += yl[i+16] * (qs[i/2] & 0x0030);\n                acc2[2] += yl[i+17] * (qs[i/2] & 0x3000);\n                acc1[3] += yl[i+24] * (qs[i/2] & 0x00c0);\n                acc2[3] += yl[i+25] * (qs[i/2] & 0xc000);\n            }\n\n            float dall = dh[0];\n            float dmin = dh[1];\n            sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * 1.f/ 1.f +\n                                 (acc1[1] + 1.f/256.f * acc2[1]) * (sc[1] & 0xF) * 1.f/ 4.f +\n                                 (acc1[2] + 1.f/256.f * acc2[2]) * (sc[2] & 0xF) * 1.f/16.f +\n                                 (acc1[3] + 1.f/256.f * acc2[3]) * (sc[3] & 0xF) * 1.f/64.f) -\n                         dmin * (sumy[0] * (sc[0] >> 4) + sumy[1] * (sc[1] >> 4) + sumy[2] * (sc[2] >> 4) + sumy[3] * (sc[3] >> 4));\n\n            qs += step/2;\n            sc += step;\n            dh += step/2;\n        }\n\n        y4 += 16 * QK_K;\n    }\n#endif\n\n    for (int row = 0; row < N_DST; ++row) {\n        all_sum = simd_sum(sumf[row]);\n        if (tiisg == 0) {\n            dst[r1*ne0 + first_row + row] = all_sum;\n        }\n    }\n}\n\n#if QK_K == 256\nkernel void kernel_mul_mat_q3_K_f32(\n        device const  void * src0,\n        device const float * src1,\n        device       float * dst,\n        constant   int64_t & ne00,\n        constant   int64_t & ne10,\n        constant   int64_t & ne0,\n        constant   int64_t & ne1,\n        uint2 tgpig[[threadgroup_position_in_grid]],\n        uint tiisg[[thread_index_in_simdgroup]],\n        uint sgitg[[simdgroup_index_in_threadgroup]]) {\n\n    const int nb = ne00/QK_K;\n\n    const int64_t r0 = tgpig.x;\n    const int64_t r1 = tgpig.y;\n\n    const int first_row = (r0 * N_SIMDGROUP + sgitg) * 2;\n\n    device const block_q3_K * x = (device const block_q3_K *) src0 + first_row*nb;\n    device const float     * yy = (device const float      *) src1 + r1*ne10;\n\n    float yl[16];\n\n    const uint16_t kmask1 = 0x0303;\n    const uint16_t kmask2 = 0x0f0f;\n\n    const int tid = tiisg/2;\n    const int ix  = tiisg%2;\n    const int ip  = tid/8;          // 0 or 1\n    const int il  = tid/2 - 4*ip;   // 0...3\n    const int ir  = tid%2;\n    const int n   = 8;\n    const int l0  = n*ir;\n\n    const uint16_t m1 = 1 << (4*ip + il);\n    const uint16_t m2 = m1 << 8;\n\n    const int shift = 2*il;\n    const uint16_t qm1 = 0x0003 << shift;\n    const uint16_t qm2 = 0x0300 << shift;\n    const int32_t v1 = 4 << shift;\n    const int32_t v2 = 1024 << shift;\n\n    const uint16_t s_shift1 = 4*ip;\n    const uint16_t s_shift2 = s_shift1 + 2*(il/2);\n    const int ik = 4 + (il%2);\n\n    const int q_offset = 32*ip + l0;\n    const int y_offset = 128*ip + 32*il + l0;\n\n    const int step = sizeof(block_q3_K) * nb / 2;\n\n    device const float * y1 = yy + ix*QK_K + y_offset;\n\n    float sumf1[2] = {0.f}, sumf2[2] = {0.f};\n    for (int i = ix; i < nb; i += 2) {\n\n        for (int l = 0; l < 8; ++l) {\n            yl[l+0] = y1[l+ 0];\n            yl[l+8] = y1[l+16];\n        }\n\n        device const uint16_t * q = (device const uint16_t *)(x[i].qs + q_offset);\n        device const uint16_t * h = (device const uint16_t *)(x[i].hmask + l0);\n        device const uint16_t * a = (device const uint16_t *)(x[i].scales);\n        device const half * dh = &x[i].d;\n\n        for (int row = 0; row < 2; ++row) {\n\n            const float d_all = (float)dh[0];\n            const char2 scales = as_type<char2>((uint16_t)(((a[il] >> s_shift1) & kmask2) | (((a[ik] >> s_shift2) & kmask1) << 4)));\n\n            float s1 = 0, s2 = 0;\n            for (int l = 0; l < n; l += 2) {\n                const uint16_t qs = q[l/2];\n                s1 += yl[l+0] * ((int32_t)(qs & qm1) - ((h[l/2] & m1) ? 0 : v1));\n                s2 += yl[l+1] * ((int32_t)(qs & qm2) - ((h[l/2] & m2) ? 0 : v2));\n            }\n            float d = d_all * (s1 + 1.f/256.f * s2);\n            sumf1[row] += d * scales[0];\n            sumf2[row] += d;\n\n            s1 = s2 = 0;\n            for (int l = 0; l < n; l += 2) {\n                const uint16_t qs = q[l/2+8];\n                s1 += yl[l+8] * ((int32_t)(qs & qm1) - ((h[l/2+8] & m1) ? 0 : v1));\n                s2 += yl[l+9] * ((int32_t)(qs & qm2) - ((h[l/2+8] & m2) ? 0 : v2));\n            }\n            d = d_all * (s1 + 1.f/256.f * s2);\n            sumf1[row] += d * scales[1];\n            sumf2[row] += d;\n\n            q  += step;\n            h  += step;\n            a  += step;\n            dh += step;\n\n        }\n\n        y1 += 2 * QK_K;\n\n    }\n\n    for (int row = 0; row < 2; ++row) {\n        const float sumf = (sumf1[row] - 32.f*sumf2[row]) / (1 << shift);\n        const float tot = simd_sum(sumf);\n        if (tiisg == 0) {\n            dst[r1*ne0 + first_row + row] = tot;\n        }\n    }\n}\n#else\nkernel void kernel_mul_mat_q3_K_f32(\n        device const  void * src0,\n        device const float * src1,\n        device       float * dst,\n        constant   int64_t & ne00,\n        constant   int64_t & ne10,\n        constant   int64_t & ne0,\n        constant   int64_t & ne1,\n        uint2 tgpig[[threadgroup_position_in_grid]],\n        uint tiisg[[thread_index_in_simdgroup]],\n        uint sgitg[[simdgroup_index_in_threadgroup]]) {\n\n    const int nb = ne00/QK_K;\n\n    const int64_t r0 = tgpig.x;\n    const int64_t r1 = tgpig.y;\n\n    const int row = 2 * r0 + sgitg;\n\n    device const block_q3_K * x = (device const block_q3_K *) src0 + row*nb;\n    device const float     * yy = (device const float      *) src1 + r1*ne10;\n    const int ix = tiisg/4;\n    const int il = 4 * (tiisg%4);// 0, 4, 8, 12\n    const int im = il/8;         // 0, 0, 1, 1\n    const int in = il%8;         // 0, 4, 0, 4\n\n    float2 sum = {0.f, 0.f};\n\n    for (int i = ix; i < nb; i += 8) {\n\n        const float d_all = (float)(x[i].d);\n\n        device const uint16_t * q = (device const uint16_t *)(x[i].qs + il);\n        device const uint16_t * h = (device const uint16_t *)(x[i].hmask + in);\n        device const uint16_t * s = (device const uint16_t *)(x[i].scales);\n        device const float    * y = yy + i * QK_K + il;\n\n        const float d1 = d_all * ((int32_t)(s[0] & 0x000F) - 8);\n        const float d2 = d_all * ((int32_t)(s[0] & 0x00F0) - 128) * 1.f/64.f;\n        const float d3 = d_all * ((int32_t)(s[0] & 0x0F00) - 2048) * 1.f/4096.f;\n        const float d4 = d_all * ((int32_t)(s[0] & 0xF000) - 32768) * 1.f/262144.f;\n\n        for (int l = 0; l < 4; l += 2) {\n            const uint16_t hm = h[l/2] >> im;\n            sum[0] += y[l+ 0] * d1 * ((int32_t)(q[l/2] & 0x0003) - ((hm & 0x0001) ? 0 :  4))\n                    + y[l+16] * d2 * ((int32_t)(q[l/2] & 0x000c) - ((hm & 0x0004) ? 0 : 16))\n                    + y[l+32] * d3 * ((int32_t)(q[l/2] & 0x0030) - ((hm & 0x0010) ? 0 : 64))\n                    + y[l+48] * d4 * ((int32_t)(q[l/2] & 0x00c0) - ((hm & 0x0040) ? 0 : 256));\n            sum[1] += y[l+ 1] * d1 * ((int32_t)(q[l/2] & 0x0300) - ((hm & 0x0100) ? 0 : 1024))\n                    + y[l+17] * d2 * ((int32_t)(q[l/2] & 0x0c00) - ((hm & 0x0400) ? 0 : 4096))\n                    + y[l+33] * d3 * ((int32_t)(q[l/2] & 0x3000) - ((hm & 0x1000) ? 0 : 16384))\n                    + y[l+49] * d4 * ((int32_t)(q[l/2] & 0xc000) - ((hm & 0x4000) ? 0 : 65536));\n        }\n\n    }\n    const float sumf = sum[0] + sum[1] * 1.f/256.f;\n\n    const float tot = simd_sum(sumf);\n    if (tiisg == 0) {\n        dst[r1*ne0 + row] = tot;\n    }\n\n}\n#endif\n\n#if QK_K == 256\nkernel void kernel_mul_mat_q4_K_f32(\n        device const  void * src0,\n        device const float * src1,\n        device       float * dst,\n        constant   int64_t & ne00,\n        constant   int64_t & ne10,\n        constant   int64_t & ne0,\n        constant   int64_t & ne01[[buffer(4)]],\n        uint2 tgpig[[threadgroup_position_in_grid]],\n        uint tiisg[[thread_index_in_simdgroup]],\n        uint sgitg[[simdgroup_index_in_threadgroup]]) {\n\n    const uint16_t kmask1 = 0x3f3f;\n    const uint16_t kmask2 = 0x0f0f;\n    const uint16_t kmask3 = 0xc0c0;\n\n    const int ix = tiisg/8;  // 0...3\n    const int it = tiisg%8;  // 0...7\n    const int im = it/4;     // 0 or 1\n    const int ir = it%4;     // 0...3\n\n    const int nb = ne00/QK_K;\n    const int r0 = tgpig.x;\n    const int r1 = tgpig.y;\n    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;\n    const int ib_row = first_row * nb;\n    device const block_q4_K * x = (device const block_q4_K *) src0 + ib_row;\n    device const float      * y = (device const float      *) src1 + r1*ne10;\n    float yl[16];\n    float yh[16];\n    float sumf[N_DST]={0.f}, all_sum;\n\n    const int step = sizeof(block_q4_K) * nb / 2;\n\n    device const float * y4 = y + ix * QK_K + 64 * im + 8 * ir;\n\n    uint16_t sc16[4];\n    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;\n\n    for (int ib = ix; ib < nb; ib += 4) {\n\n        float4 sumy = {0.f, 0.f, 0.f, 0.f};\n        for (int i = 0; i < 8; ++i) {\n            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];\n            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];\n            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];\n            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];\n        }\n\n        device const uint16_t * sc = (device const uint16_t *)x[ib].scales + im;\n        device const uint16_t * q1 = (device const uint16_t *)x[ib].qs + 16 * im + 4 * ir;\n        device const half     * dh = &x[ib].d;\n\n        for (int row = 0; row < N_DST; row++) {\n\n            sc16[0] = sc[0] & kmask1;\n            sc16[1] = sc[2] & kmask1;\n            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);\n            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);\n\n            device const uint16_t * q2 = q1 + 32;\n\n            float4 acc1 = {0.f, 0.f, 0.f, 0.f};\n            float4 acc2 = {0.f, 0.f, 0.f, 0.f};\n            for (int i = 0; i < 8; i += 2) {\n                acc1[0] += yl[i+0] * (q1[i/2] & 0x000F);\n                acc1[1] += yl[i+1] * (q1[i/2] & 0x0F00);\n                acc1[2] += yl[i+8] * (q1[i/2] & 0x00F0);\n                acc1[3] += yl[i+9] * (q1[i/2] & 0xF000);\n                acc2[0] += yh[i+0] * (q2[i/2] & 0x000F);\n                acc2[1] += yh[i+1] * (q2[i/2] & 0x0F00);\n                acc2[2] += yh[i+8] * (q2[i/2] & 0x00F0);\n                acc2[3] += yh[i+9] * (q2[i/2] & 0xF000);\n            }\n\n            float dall = dh[0];\n            float dmin = dh[1];\n            sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +\n                                 (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +\n                                 (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +\n                                 (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -\n                         dmin * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);\n\n            q1 += step;\n            sc += step;\n            dh += step;\n        }\n\n        y4 += 4 * QK_K;\n    }\n\n    for (int row = 0; row < N_DST; ++row) {\n        all_sum = simd_sum(sumf[row]);\n        if (tiisg == 0) {\n            dst[r1*ne0 + first_row + row] = all_sum;\n        }\n    }\n}\n#else\nkernel void kernel_mul_mat_q4_K_f32(\n        device const  void * src0,\n        device const float * src1,\n        device       float * dst,\n        constant   int64_t & ne00,\n        constant   int64_t & ne10,\n        constant   int64_t & ne0,\n        constant   int64_t & ne01[[buffer(4)]],\n        uint2 tgpig[[threadgroup_position_in_grid]],\n        uint tiisg[[thread_index_in_simdgroup]],\n        uint sgitg[[simdgroup_index_in_threadgroup]]) {\n\n    const int ix = tiisg/4;  // 0...7\n    const int it = tiisg%4;  // 0...3\n\n    const int nb = ne00/QK_K;\n    const int r0 = tgpig.x;\n    const int r1 = tgpig.y;\n    const int first_row = (r0 * N_SIMDGROUP + sgitg) * N_DST;\n    const int ib_row = first_row * nb;\n    device const block_q4_K * x = (device const block_q4_K *) src0 + ib_row;\n    device const float      * y = (device const float      *) src1 + r1*ne10;\n    float yl[8];\n    float yh[8];\n    float sumf[N_DST]={0.f}, all_sum;\n\n    const int step = sizeof(block_q4_K) * nb / 2;\n\n    device const float * y4 = y + ix * QK_K + 8 * it;\n\n    uint16_t sc16[4];\n\n    for (int ib = ix; ib < nb; ib += 8) {\n\n        float2 sumy = {0.f, 0.f};\n        for (int i = 0; i < 8; ++i) {\n            yl[i] = y4[i+ 0]; sumy[0] += yl[i];\n            yh[i] = y4[i+32]; sumy[1] += yh[i];\n        }\n\n        device const uint16_t * sc = (device const uint16_t *)x[ib].scales;\n        device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 4 * it;\n        device const half     * dh = x[ib].d;\n\n        for (int row = 0; row < N_DST; row++) {\n\n            sc16[0] = sc[0] & 0x000f;\n            sc16[1] = sc[0] & 0x0f00;\n            sc16[2] = sc[0] & 0x00f0;\n            sc16[3] = sc[0] & 0xf000;\n\n            float2 acc1 = {0.f, 0.f};\n            float2 acc2 = {0.f, 0.f};\n            for (int i = 0; i < 8; i += 2) {\n                acc1[0] += yl[i+0] * (qs[i/2] & 0x000F);\n                acc1[1] += yl[i+1] * (qs[i/2] & 0x0F00);\n                acc2[0] += yh[i+0] * (qs[i/2] & 0x00F0);\n                acc2[1] += yh[i+1] * (qs[i/2] & 0xF000);\n            }\n\n            float dall = dh[0];\n            float dmin = dh[1];\n            sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc1[1]) * sc16[0] +\n                                 (acc2[0] + 1.f/256.f * acc2[1]) * sc16[1] * 1.f/4096.f) -\n                         dmin * 1.f/16.f * (sumy[0] * sc16[2] + sumy[1] * sc16[3] * 1.f/256.f);\n\n            qs += step;\n            sc += step;\n            dh += step;\n        }\n\n        y4 += 8 * QK_K;\n    }\n\n    for (int row = 0; row < N_DST; ++row) {\n        all_sum = simd_sum(sumf[row]);\n        if (tiisg == 0) {\n            dst[r1*ne0 + first_row + row] = all_sum;\n        }\n    }\n}\n#endif\n\nkernel void kernel_mul_mat_q5_K_f32(\n        device const  void * src0,\n        device const float * src1,\n        device       float * dst,\n        constant   int64_t & ne00,\n        constant   int64_t & ne10,\n        constant   int64_t & ne0,\n        uint2 tgpig[[threadgroup_position_in_grid]],\n        uint tiisg[[thread_index_in_simdgroup]],\n        uint sgitg[[simdgroup_index_in_threadgroup]]) {\n\n    const int nb = ne00/QK_K;\n\n    const int64_t r0 = tgpig.x;\n    const int64_t r1 = tgpig.y;\n\n    const int first_row = (r0 * N_SIMDGROUP + sgitg) * 2;\n\n    device const block_q5_K * x = (device const block_q5_K *) src0 + first_row*nb;\n    device const float     * yy = (device const float      *) src1 + r1*ne10;\n\n    float sumf[2]={0.f};\n\n    const int step = sizeof(block_q5_K) * nb;\n\n#if QK_K == 256\n#\n    float yl[16], yh[16];\n\n    const uint16_t kmask1 = 0x3f3f;\n    const uint16_t kmask2 = 0x0f0f;\n    const uint16_t kmask3 = 0xc0c0;\n\n    const int tid = tiisg/4;\n    const int ix  = tiisg%4;\n    const int im  = tid/4;\n    const int ir  = tid%4;\n    const int n   = 8;\n\n    const int l0 = n*ir;\n    const int q_offset = 32*im + l0;\n    const int y_offset = 64*im + l0;\n\n    const uint8_t hm1 = 1u << (2*im);\n    const uint8_t hm2 = hm1 << 1;\n    const uint8_t hm3 = hm1 << 4;\n    const uint8_t hm4 = hm2 << 4;\n\n    uint16_t sc16[4];\n    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;\n\n    device const float * y1 = yy + ix*QK_K + y_offset;\n\n    for (int i = ix; i < nb; i += 4) {\n\n        device const uint8_t * q1 = x[i].qs + q_offset;\n        device const uint8_t * qh = x[i].qh + l0;\n        device const half * dh = &x[i].d;\n        device const uint16_t * a = (device const uint16_t *)x[i].scales + im;\n\n        device const float * y2 = y1 + 128;\n        float4 sumy = {0.f, 0.f, 0.f, 0.f};\n        for (int l = 0; l < 8; ++l) {\n            yl[l+0] = y1[l+ 0]; sumy[0] += yl[l+0];\n            yl[l+8] = y1[l+32]; sumy[1] += yl[l+8];\n            yh[l+0] = y2[l+ 0]; sumy[2] += yh[l+0];\n            yh[l+8] = y2[l+32]; sumy[3] += yh[l+8];\n        }\n\n        for (int row = 0; row < 2; ++row) {\n\n            device const uint8_t * q2 = q1 + 64;\n\n            sc16[0] = a[0] & kmask1;\n            sc16[1] = a[2] & kmask1;\n            sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);\n            sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);\n\n            float4 acc = {0.f, 0.f, 0.f, 0.f};\n            for (int l = 0; l < n; ++l) {\n                uint8_t h = qh[l];\n                acc[0] += yl[l+0] * ((uint16_t)(q1[l] & 0x0F) + (h & hm1 ? 16 : 0));\n                acc[1] += yl[l+8] * ((uint16_t)(q1[l] & 0xF0) + (h & hm2 ? 256 : 0));\n                acc[2] += yh[l+0] * ((uint16_t)(q2[l] & 0x0F) + (h & hm3 ? 16 : 0));\n                acc[3] += yh[l+8] * ((uint16_t)(q2[l] & 0xF0) + (h & hm4 ? 256 : 0));\n            }\n            const float dall = dh[0];\n            const float dmin = dh[1];\n            sumf[row] += dall * (acc[0] * sc8[0] + acc[1] * sc8[1] * 1.f/16.f + acc[2] * sc8[4] + acc[3] * sc8[5] * 1.f/16.f) -\n                         dmin * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);\n\n            q1 += step;\n            qh += step;\n            dh += step/2;\n            a  += step/2;\n\n        }\n\n        y1 += 4 * QK_K;\n\n    }\n#else\n    float yl[8], yh[8];\n\n    const int il = 4 * (tiisg/8);  // 0, 4, 8, 12\n    const int ix = tiisg%8;\n    const int im = il/8;         // 0, 0, 1, 1\n    const int in = il%8;         // 0, 4, 0, 4\n\n    device const float * y = yy + ix*QK_K + il;\n\n    for (int i = ix; i < nb; i += 8) {\n\n        for (int l = 0; l < 4; ++l) {\n            yl[l+0] = y[l+ 0];\n            yl[l+4] = y[l+16];\n            yh[l+0] = y[l+32];\n            yh[l+4] = y[l+48];\n        }\n\n        device const half * dh = &x[i].d;\n        device const uint8_t * q = x[i].qs + il;\n        device const uint8_t * h = x[i].qh + in;\n        device const int8_t  * s = x[i].scales;\n\n        for (int row = 0; row < 2; ++row) {\n\n            const float d = dh[0];\n\n            float2 acc = {0.f, 0.f};\n            for (int l = 0; l < 4; ++l) {\n                const uint8_t hl = h[l] >> im;\n                acc[0] += yl[l+0] * s[0] * ((int16_t)(q[l+ 0] & 0x0F) - (hl & 0x01 ? 0 : 16))\n                        + yl[l+4] * s[1] * ((int16_t)(q[l+16] & 0x0F) - (hl & 0x04 ? 0 : 16));\n                acc[1] += yh[l+0] * s[2] * ((int16_t)(q[l+ 0] & 0xF0) - (hl & 0x10 ? 0 : 256))\n                        + yh[l+4] * s[3] * ((int16_t)(q[l+16] & 0xF0) - (hl & 0x40 ? 0 : 256));\n            }\n            sumf[row] += d * (acc[0] + 1.f/16.f * acc[1]);\n\n            q += step;\n            h += step;\n            s += step;\n            dh += step/2;\n\n        }\n\n        y += 8 * QK_K;\n    }\n#endif\n\n    for (int row = 0; row < 2; ++row) {\n        const float tot = simd_sum(sumf[row]);\n        if (tiisg == 0) {\n            dst[r1*ne0 + first_row + row] = tot;\n        }\n    }\n\n}\n\nkernel void kernel_mul_mat_q6_K_f32(\n        device const  void * src0,\n        device const float * src1,\n        device       float * dst,\n        constant   int64_t & ne00,\n        constant   int64_t & ne10,\n        constant   int64_t & ne0,\n        uint2 tgpig[[threadgroup_position_in_grid]],\n        uint tiisg[[thread_index_in_simdgroup]],\n        uint sgitg[[simdgroup_index_in_threadgroup]]) {\n\n    const uint8_t kmask1 = 0x03;\n    const uint8_t kmask2 = 0x0C;\n    const uint8_t kmask3 = 0x30;\n    const uint8_t kmask4 = 0xC0;\n\n    const int nb = ne00/QK_K;\n\n    const int64_t r0 = tgpig.x;\n    const int64_t r1 = tgpig.y;\n\n    const int row = 2 * r0 + sgitg;\n\n    device const block_q6_K * x = (device const block_q6_K *) src0 + row * nb; //r0*nb;\n    device const float     * yy = (device const float      *) src1 + r1*ne10;\n\n    float sumf = 0;\n\n#if QK_K == 256\n    const int tid  = tiisg/2;\n    const int ix   = tiisg%2;\n    const int ip   = tid/8;         // 0 or 1\n    const int il   = tid%8;\n    const int n    = 4;\n    const int l0   = n*il;\n    const int is   = 8*ip + l0/16;\n\n    const int y_offset = 128*ip + l0;\n    const int q_offset_l = 64*ip + l0;\n    const int q_offset_h = 32*ip + l0;\n\n    for (int i = ix; i < nb; i += 2) {\n\n        device const uint8_t * q1 = x[i].ql + q_offset_l;\n        device const uint8_t * q2 = q1 + 32;\n        device const uint8_t * qh = x[i].qh + q_offset_h;\n        device const int8_t  * sc = x[i].scales + is;\n\n        device const float * y = yy + i * QK_K + y_offset;\n\n        const float dall = x[i].d;\n\n        float4 sums = {0.f, 0.f, 0.f, 0.f};\n        for (int l = 0; l < n; ++l) {\n            sums[0] += y[l+ 0] * ((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);\n            sums[1] += y[l+32] * ((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);\n            sums[2] += y[l+64] * ((int8_t)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);\n            sums[3] += y[l+96] * ((int8_t)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);\n        }\n\n        sumf += dall * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);\n\n    }\n\n#else\n    const int ix  = tiisg/4;\n    const int il  = 4*(tiisg%4);\n\n    for (int i = ix; i < nb; i += 8) {\n        device const float * y = yy + i * QK_K + il;\n        device const uint8_t * ql = x[i].ql + il;\n        device const uint8_t * qh = x[i].qh + il;\n        device const int8_t  * s  = x[i].scales;\n\n        const float d = x[i].d;\n\n        float4 sums = {0.f, 0.f, 0.f, 0.f};\n        for (int l = 0; l < 4; ++l) {\n            sums[0] += y[l+ 0] * ((int8_t)((ql[l+ 0] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);\n            sums[1] += y[l+16] * ((int8_t)((ql[l+16] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);\n            sums[2] += y[l+32] * ((int8_t)((ql[l+ 0] >>  4) | ((qh[l] & kmask3) >> 0)) - 32);\n            sums[3] += y[l+48] * ((int8_t)((ql[l+16] >>  4) | ((qh[l] & kmask4) >> 2)) - 32);\n        }\n        sumf += d * (sums[0] * s[0] + sums[1] * s[1] + sums[2] * s[2] + sums[3] * s[3]);\n    }\n\n#endif\n\n    const float tot = simd_sum(sumf);\n    if (tiisg == 0) {\n        dst[r1*ne0 + row] = tot;\n    }\n}\n";

        if (error) {
            fprintf(stderr, "%s: error: %s\n", __func__, [[error description] UTF8String]);
            exit(1);
        }

#ifdef GGML_QKK_64
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.preprocessorMacros = @{ @"QK_K" : @(64) };
        ctx->library = [ctx->device newLibraryWithSource:src options:options error:&error];
#else
        ctx->library = [ctx->device newLibraryWithSource:src options:nil error:&error];
#endif
        if (error) {
            fprintf(stderr, "%s: error: %s\n", __func__, [[error description] UTF8String]);
            exit(1);
        }
    }
#endif

    // load kernels
    {
#define GGML_METAL_ADD_KERNEL(name) \
        ctx->function_##name = [ctx->library newFunctionWithName:@"kernel_"#name]; \
        ctx->pipeline_##name = [ctx->device newComputePipelineStateWithFunction:ctx->function_##name error:nil]; \
        fprintf(stderr, "%s: loaded %-32s %16p\n", __func__, "kernel_"#name, (void *) ctx->pipeline_##name);

        GGML_METAL_ADD_KERNEL(add);
        GGML_METAL_ADD_KERNEL(add_row);
        GGML_METAL_ADD_KERNEL(mul);
        GGML_METAL_ADD_KERNEL(mul_row);
        GGML_METAL_ADD_KERNEL(scale);
        GGML_METAL_ADD_KERNEL(silu);
        GGML_METAL_ADD_KERNEL(relu);
        GGML_METAL_ADD_KERNEL(gelu);
        GGML_METAL_ADD_KERNEL(soft_max);
        GGML_METAL_ADD_KERNEL(diag_mask_inf);
        GGML_METAL_ADD_KERNEL(get_rows_f16);
        GGML_METAL_ADD_KERNEL(get_rows_q4_0);
        GGML_METAL_ADD_KERNEL(get_rows_q4_1);
        GGML_METAL_ADD_KERNEL(get_rows_q2_K);
        GGML_METAL_ADD_KERNEL(get_rows_q3_K);
        GGML_METAL_ADD_KERNEL(get_rows_q4_K);
        GGML_METAL_ADD_KERNEL(get_rows_q5_K);
        GGML_METAL_ADD_KERNEL(get_rows_q6_K);
        GGML_METAL_ADD_KERNEL(rms_norm);
        GGML_METAL_ADD_KERNEL(norm);
        GGML_METAL_ADD_KERNEL(mul_mat_f16_f32);
        GGML_METAL_ADD_KERNEL(mul_mat_q4_0_f32);
        GGML_METAL_ADD_KERNEL(mul_mat_q4_1_f32);
        GGML_METAL_ADD_KERNEL(mul_mat_q2_K_f32);
        GGML_METAL_ADD_KERNEL(mul_mat_q3_K_f32);
        GGML_METAL_ADD_KERNEL(mul_mat_q4_K_f32);
        GGML_METAL_ADD_KERNEL(mul_mat_q5_K_f32);
        GGML_METAL_ADD_KERNEL(mul_mat_q6_K_f32);
        GGML_METAL_ADD_KERNEL(rope);
        GGML_METAL_ADD_KERNEL(alibi_f32);
        GGML_METAL_ADD_KERNEL(cpy_f32_f16);
        GGML_METAL_ADD_KERNEL(cpy_f32_f32);
        GGML_METAL_ADD_KERNEL(cpy_f16_f16);

#undef GGML_METAL_ADD_KERNEL
    }

    fprintf(stderr, "%s: recommendedMaxWorkingSetSize = %8.2f MB\n", __func__, ctx->device.recommendedMaxWorkingSetSize / 1024.0 / 1024.0);
    fprintf(stderr, "%s: hasUnifiedMemory             = %s\n",       __func__, ctx->device.hasUnifiedMemory ? "true" : "false");
    if (ctx->device.maxTransferRate != 0) {
        fprintf(stderr, "%s: maxTransferRate              = %8.2f MB/s\n", __func__, ctx->device.maxTransferRate / 1024.0 / 1024.0);
    } else {
        fprintf(stderr, "%s: maxTransferRate              = built-in GPU\n", __func__);
    }

    return ctx;
}

void ggml_metal_free(struct ggml_metal_context * ctx) {
    fprintf(stderr, "%s: deallocating\n", __func__);
    for (int i = 0; i < ctx->n_buffers; ++i) {
        [ctx->buffers[i].metal release];
    }
    free(ctx);
}

void ggml_metal_set_n_cb(struct ggml_metal_context * ctx, int n_cb) {
    ctx->n_cb = n_cb;
}

bool ggml_metal_if_optimized(struct ggml_metal_context * ctx) {
    if (ctx->concur_list_len) {
        return true;
    }
    return false;
}

// finds the Metal buffer that contains the tensor data on the GPU device
// the assumption is that there is 1-to-1 mapping between the host and device memory buffers, so we can find the
// Metal buffer based on the host memory pointer
//
static id<MTLBuffer> ggml_metal_get_buffer(struct ggml_metal_context * ctx, struct ggml_tensor * t, size_t * offs) {
    //fprintf(stderr, "%s: data tensor '%16s', offs_data = %8ld, offs_eval = %8ld, offs_cach = %8ld\n", __func__, t->name, offs_data, offs_eval, offs_cach);

    const int64_t tsize = ggml_nbytes(t);

    // find the view that contains the tensor fully
    for (int i = 0; i < ctx->n_buffers; ++i) {
        const int64_t ioffs = (int64_t) t->data - (int64_t) ctx->buffers[i].data;

        if (ioffs >= 0 && ioffs + tsize <= (int64_t) ctx->buffers[i].size) {
            *offs = (size_t) ioffs;

            //fprintf(stderr, "%s: '%s' tensor '%16s', offs = %8ld\n", __func__, ctx->buffers[i].name, t->name, *offs);

            return ctx->buffers[i].metal;
        }
    }

    fprintf(stderr, "%s: error: buffer is nil\n", __func__);

    return nil;
}

bool ggml_metal_add_buffer(
        struct ggml_metal_context * ctx,
                     const char * name,
                           void * data,
                         size_t   size,
                         size_t   max_size) {
    if (ctx->n_buffers >= GGML_METAL_MAX_BUFFERS) {
        fprintf(stderr, "%s: too many buffers\n", __func__);
        return false;
    }

    if (data) {
        // verify that the buffer does not overlap with any of the existing buffers
        for (int i = 0; i < ctx->n_buffers; ++i) {
            const int64_t ioffs = (int64_t) data - (int64_t) ctx->buffers[i].data;

            if (ioffs >= 0 && ioffs < (int64_t) ctx->buffers[i].size) {
                fprintf(stderr, "%s: error: buffer '%s' overlaps with '%s'\n", __func__, name, ctx->buffers[i].name);
                return false;
            }
        }

        const size_t size_page = getpagesize();

        size_t size_aligned = size;
        if ((size_aligned % size_page) != 0) {
            size_aligned += (size_page - (size_aligned % size_page));
        }

        // the buffer fits into the max buffer size allowed by the device
        if (size_aligned <= ctx->device.maxBufferLength) {
            ctx->buffers[ctx->n_buffers].name = name;
            ctx->buffers[ctx->n_buffers].data = data;
            ctx->buffers[ctx->n_buffers].size = size;

            ctx->buffers[ctx->n_buffers].metal = [ctx->device newBufferWithBytesNoCopy:data length:size_aligned options:MTLResourceStorageModeShared deallocator:nil];

            if (ctx->buffers[ctx->n_buffers].metal == nil) {
                fprintf(stderr, "%s: failed to allocate '%-16s' buffer, size = %8.2f MB\n", __func__, name, size_aligned / 1024.0 / 1024.0);
                return false;
            }

            fprintf(stderr, "%s: allocated '%-16s' buffer, size = %8.2f MB", __func__, name, size_aligned / 1024.0 / 1024.0);

            ++ctx->n_buffers;
        } else {
            // this overlap between the views will guarantee that the tensor with the maximum size will fully fit into
            // one of the views
            const size_t size_ovlp = ((max_size + size_page - 1) / size_page + 1) * size_page; // round-up 2 pages just in case
            const size_t size_step = ctx->device.maxBufferLength - size_ovlp;
            const size_t size_view = ctx->device.maxBufferLength;

            for (size_t i = 0; i < size; i += size_step) {
                const size_t size_step_aligned = (i + size_view <= size) ? size_view : (size_aligned - i);

                ctx->buffers[ctx->n_buffers].name = name;
                ctx->buffers[ctx->n_buffers].data = (void *) ((uint8_t *) data + i);
                ctx->buffers[ctx->n_buffers].size = size_step_aligned;

                ctx->buffers[ctx->n_buffers].metal = [ctx->device newBufferWithBytesNoCopy:(void *) ((uint8_t *) data + i) length:size_step_aligned options:MTLResourceStorageModeShared deallocator:nil];

                if (ctx->buffers[ctx->n_buffers].metal == nil) {
                    fprintf(stderr, "%s: failed to allocate '%-16s' buffer, size = %8.2f MB\n", __func__, name, size_step_aligned / 1024.0 / 1024.0);
                    return false;
                }

                fprintf(stderr, "%s: allocated '%-16s' buffer, size = %8.2f MB, offs = %12ld", __func__, name, size_step_aligned / 1024.0 / 1024.0, i);
                if (i + size_step < size) {
                    fprintf(stderr, "\n");
                }

                ++ctx->n_buffers;
            }
        }

        fprintf(stderr, ", (%8.2f / %8.2f)",
                ctx->device.currentAllocatedSize / 1024.0 / 1024.0,
                ctx->device.recommendedMaxWorkingSetSize / 1024.0 / 1024.0);

        if (ctx->device.currentAllocatedSize > ctx->device.recommendedMaxWorkingSetSize) {
            fprintf(stderr, ", warning: current allocated size is greater than the recommended max working set size\n");
        } else {
            fprintf(stderr, "\n");
        }
    }

    return true;
}

void ggml_metal_set_tensor(
        struct ggml_metal_context * ctx,
        struct ggml_tensor * t) {
    metal_printf("%s: set input for tensor '%s'\n", __func__, t->name);

    size_t offs;
    id<MTLBuffer> id_dst = ggml_metal_get_buffer(ctx, t, &offs);

    memcpy((void *) ((uint8_t *) id_dst.contents + offs), t->data, ggml_nbytes(t));
}

void ggml_metal_get_tensor(
        struct ggml_metal_context * ctx,
        struct ggml_tensor * t) {
    metal_printf("%s: extract results for tensor '%s'\n", __func__, t->name);

    size_t offs;
    id<MTLBuffer> id_src = ggml_metal_get_buffer(ctx, t, &offs);

    memcpy(t->data, (void *) ((uint8_t *) id_src.contents + offs), ggml_nbytes(t));
}

void ggml_metal_graph_find_concurrency(
        struct ggml_metal_context * ctx,
        struct ggml_cgraph * gf) {
    int search_depth = gf->n_nodes; //we only find concurrency in this range to avoid wasting too much time
    int nodes_unused[GGML_MAX_NODES];

    for (int i = 0; i < GGML_MAX_NODES; i++) {ctx->concur_list[i] = 0;}
    for (int i = 0; i < gf->n_nodes; i++) {nodes_unused[i] = 1;}
    ctx->concur_list_len = 0;

    int n_left = gf->n_nodes;
    int n_start = 0; // all nodes before n_start at nodes_unused array have been sorted and store back to ctx->concur_list
    int level_pos = 0;  // at ctx->concur_list, the last layer (level) ends at level_pos

    while (n_left > 0) {
        // number of nodes at a layer (that can be issued concurrently)
        int concurrency = 0;
        for (int i = n_start; i < ((n_start + search_depth > gf->n_nodes) ? gf->n_nodes : n_start + search_depth); i++) {
            if (nodes_unused[i]) {
                // if the requirements for gf->nodes[i] are satisfied
                int exe_flag=1;
                // scan all srcs
                for (int src_ind = 0; src_ind < GGML_MAX_SRC; src_ind++) {
                    struct ggml_tensor * src_cur = gf->nodes[i]->src[src_ind];
                    if (src_cur) {
                        // if is leaf nodes it's satisfied.
                        if (src_cur->op == GGML_OP_NONE && src_cur->grad == NULL) {continue;}

                        // otherwise this src should be the output from previous nodes.
                        int is_found = 0;
                        // scan 2*search_depth back because we inserted barrier.
                        for (int j = ((level_pos - 2*search_depth) < 0 ? 0 : (level_pos - 2*search_depth)); j < level_pos; j++) {
                            if (gf->nodes[ctx->concur_list[j]] == src_cur) {is_found = 1; break;}
                        }
                        if (is_found == 0) {exe_flag = 0; break;}
                    }
                }
                if (exe_flag) {
                    // check if nodes[i]'s data will be overwritten by a node before nodes[i].
                    // if node[5] and node[3] write to the same memory region, then we can't issue node[5] before node[3]
                    int64_t data_start = (int64_t) gf->nodes[i]->data;
                    int64_t length = (int64_t) ggml_nbytes(gf->nodes[i]);
                    for (int j = n_start; j < i; j++) {
                        if (nodes_unused[j] && gf->nodes[j]->op != GGML_OP_RESHAPE \
                                            && gf->nodes[j]->op != GGML_OP_VIEW \
                                            && gf->nodes[j]->op != GGML_OP_TRANSPOSE \
                                            && gf->nodes[j]->op != GGML_OP_PERMUTE) {
                            if (((int64_t)gf->nodes[j]->data) >= data_start + length || \
                                ((int64_t)gf->nodes[j]->data) + (int64_t) ggml_nbytes(gf->nodes[j]) <= data_start) {
                                continue;
                            } else {
                                exe_flag = 0;
                            }
                        }
                    }
                }
                if (exe_flag) {
                    ctx->concur_list[level_pos + concurrency] = i;
                    nodes_unused[i] = 0;
                    concurrency++;
                    ctx->concur_list_len++;
                }
            }
        }
        n_left -= concurrency;
        // adding a barrier different layer
        ctx->concur_list[level_pos + concurrency] = -1;
        ctx->concur_list_len++;
        // jump all sorted nodes at nodes_bak
        while (!nodes_unused[n_start]) {n_start++;}
        level_pos += concurrency + 1;
    }

    if (ctx->concur_list_len > GGML_MAX_NODES) {
        fprintf(stderr, "%s: too many elements for metal ctx->concur_list!\n", __func__);
    }
}

void ggml_metal_graph_compute(
        struct ggml_metal_context * ctx,
               struct ggml_cgraph * gf) {
    metal_printf("%s: evaluating graph\n", __func__);

    // if there is ctx->concur_list, dispatch concurrently
    // else fallback to serial dispatch
    MTLComputePassDescriptor * edesc = MTLComputePassDescriptor.computePassDescriptor;

    const bool has_concur = ctx->concur_list_len && ctx->concur_list_len <= GGML_MAX_NODES;

    const int n_nodes  = has_concur ? ctx->concur_list_len      : gf->n_nodes;
    edesc.dispatchType = has_concur ? MTLDispatchTypeConcurrent : MTLDispatchTypeSerial;

    // create multiple command buffers and enqueue them
    // then, we encode the graph into the command buffers in parallel

    const int n_cb = ctx->n_cb;

    NSMutableArray * command_buffers = [NSMutableArray arrayWithCapacity:n_cb];

    for (int i = 0; i < n_cb; ++i) {
        command_buffers[i] = [ctx->queue commandBuffer];

        // enqueue the command buffers in order to specify their execution order
        [command_buffers[i] enqueue];
    }

    // TODO: is this the best way to start threads?
    dispatch_queue_t queue = dispatch_queue_create("llama.cpp", DISPATCH_QUEUE_CONCURRENT);

    for (int cb_idx = 0; cb_idx < n_cb; ++cb_idx) {
        const int n_nodes_per_cb = (n_nodes + n_cb - 1) / n_cb;

        dispatch_async(queue, ^{
            size_t offs_src0 = 0;
            size_t offs_src1 = 0;
            size_t offs_dst  = 0;

            id<MTLCommandBuffer> command_buffer = command_buffers[cb_idx];

            id<MTLComputeCommandEncoder> encoder = nil;

            const int node_start =                                  (cb_idx + 0) * n_nodes_per_cb;
            const int node_end   = (cb_idx == n_cb - 1) ? n_nodes : (cb_idx + 1) * n_nodes_per_cb;

            for (int ind = node_start; ind < node_end; ++ind) {
                const int i = has_concur ? ctx->concur_list[ind] : ind;

                if (i == -1) {
                    if (encoder == nil) {
                        encoder = [command_buffer computeCommandEncoderWithDescriptor: edesc];
                        continue;
                    }
                    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    continue;
                }

                metal_printf("%s: encoding node %3d, op = %8s\n", __func__, i, ggml_op_name(gf->nodes[i]->op));

                struct ggml_tensor * src0 = gf->nodes[i]->src[0];
                struct ggml_tensor * src1 = gf->nodes[i]->src[1];
                struct ggml_tensor * dst  = gf->nodes[i];

                const int64_t  ne00 = src0 ? src0->ne[0] : 0;
                const int64_t  ne01 = src0 ? src0->ne[1] : 0;
                const int64_t  ne02 = src0 ? src0->ne[2] : 0;
                const int64_t  ne03 = src0 ? src0->ne[3] : 0;

                const uint64_t nb00 = src0 ? src0->nb[0] : 0;
                const uint64_t nb01 = src0 ? src0->nb[1] : 0;
                const uint64_t nb02 = src0 ? src0->nb[2] : 0;
                const uint64_t nb03 = src0 ? src0->nb[3] : 0;

                const int64_t  ne10 = src1 ? src1->ne[0] : 0;
                const int64_t  ne11 = src1 ? src1->ne[1] : 0;
                const int64_t  ne12 = src1 ? src1->ne[2] : 0;
                const int64_t  ne13 = src1 ? src1->ne[3] : 0; UNUSED(ne13);

                const uint64_t nb10 = src1 ? src1->nb[0] : 0;
                const uint64_t nb11 = src1 ? src1->nb[1] : 0;
                const uint64_t nb12 = src1 ? src1->nb[2] : 0;
                const uint64_t nb13 = src1 ? src1->nb[3] : 0; UNUSED(nb13);

                const int64_t  ne0  = dst ? dst->ne[0] : 0;
                const int64_t  ne1  = dst ? dst->ne[1] : 0;
                const int64_t  ne2  = dst ? dst->ne[2] : 0;
                const int64_t  ne3  = dst ? dst->ne[3] : 0;

                const uint64_t nb0  = dst ? dst->nb[0] : 0;
                const uint64_t nb1  = dst ? dst->nb[1] : 0;
                const uint64_t nb2  = dst ? dst->nb[2] : 0;
                const uint64_t nb3  = dst ? dst->nb[3] : 0;

                const enum ggml_type src0t = src0 ? src0->type : GGML_TYPE_COUNT;
                const enum ggml_type src1t = src1 ? src1->type : GGML_TYPE_COUNT;
                const enum ggml_type dstt  = dst  ? dst->type  : GGML_TYPE_COUNT;

                id<MTLBuffer> id_src0 = src0 ? ggml_metal_get_buffer(ctx, src0, &offs_src0) : nil;
                id<MTLBuffer> id_src1 = src1 ? ggml_metal_get_buffer(ctx, src1, &offs_src1) : nil;
                id<MTLBuffer> id_dst  = dst  ? ggml_metal_get_buffer(ctx, dst,  &offs_dst)  : nil;

                //metal_printf("%s: op - %s\n", __func__, ggml_op_name(dst->op));
                //if (src0) {
                //    metal_printf("%s: src0 - %4s [%5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(src0t), ne00, ne01, ne02,
                //            ggml_is_contiguous(src0), src0->name);
                //}
                //if (src1) {
                //    metal_printf("%s: src1 - %4s [%5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(src1t), ne10, ne11, ne12,
                //            ggml_is_contiguous(src1), src1->name);
                //}
                //if (dst) {
                //    metal_printf("%s: dst  - %4s [%5lld, %5lld, %5lld], 1, %s\n",  __func__, ggml_type_name(dstt),  ne0,  ne1,  ne2,
                //            dst->name);
                //}

                switch (dst->op) {
                    case GGML_OP_NONE:
                    case GGML_OP_RESHAPE:
                    case GGML_OP_VIEW:
                    case GGML_OP_TRANSPOSE:
                    case GGML_OP_PERMUTE:
                        {
                            // noop
                        } break;
                    case GGML_OP_ADD:
                        {
                            if (encoder == nil) {
                                encoder = [command_buffer computeCommandEncoderWithDescriptor: edesc];
                            }

                            if (ggml_nelements(src1) == ne10) {
                                // src1 is a row
                                [encoder setComputePipelineState:ctx->pipeline_add_row];
                            } else {
                                [encoder setComputePipelineState:ctx->pipeline_add];
                            }
                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_src1 offset:offs_src1 atIndex:1];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:2];
                            [encoder setBytes:&ne00 length:sizeof(ne00) atIndex:3];

                            const int64_t n = ggml_nelements(dst);

                            [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                        } break;
                    case GGML_OP_MUL:
                        {
                            if (encoder == nil) {
                                encoder = [command_buffer computeCommandEncoderWithDescriptor: edesc];
                            }

                            if (ggml_nelements(src1) == ne10) {
                                // src1 is a row
                                [encoder setComputePipelineState:ctx->pipeline_mul_row];
                            } else {
                                [encoder setComputePipelineState:ctx->pipeline_mul];
                            }
                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_src1 offset:offs_src1 atIndex:1];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:2];
                            [encoder setBytes:&ne00 length:sizeof(ne00) atIndex:3];

                            const int64_t n = ggml_nelements(dst);

                            [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                        } break;
                    case GGML_OP_SCALE:
                        {
                            if (encoder == nil) {
                                encoder = [command_buffer computeCommandEncoderWithDescriptor: edesc];
                            }

                            const float scale = *(const float *) src1->data;

                            [encoder setComputePipelineState:ctx->pipeline_scale];
                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                            [encoder setBytes:&scale length:sizeof(scale) atIndex:2];

                            const int64_t n = ggml_nelements(dst);

                            [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                        } break;
                    case GGML_OP_UNARY:
                        switch (ggml_get_unary_op(gf->nodes[i])) {
                            case GGML_UNARY_OP_SILU:
                                {
                                    if (encoder == nil) {
                                        encoder = [command_buffer computeCommandEncoderWithDescriptor: edesc];
                                    }

                                    [encoder setComputePipelineState:ctx->pipeline_silu];
                                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];

                                    const int64_t n = ggml_nelements(dst);

                                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                                } break;
                            case GGML_UNARY_OP_RELU:
                                {
                                    if (encoder == nil) {
                                        encoder = [command_buffer computeCommandEncoderWithDescriptor: edesc];
                                    }

                                    [encoder setComputePipelineState:ctx->pipeline_relu];
                                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];

                                    const int64_t n = ggml_nelements(dst);

                                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                                } break;
                            case GGML_UNARY_OP_GELU:
                                {
                                    if (encoder == nil) {
                                        encoder = [command_buffer computeCommandEncoderWithDescriptor: edesc];
                                    }

                                    [encoder setComputePipelineState:ctx->pipeline_gelu];
                                    [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                                    [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];

                                    const int64_t n = ggml_nelements(dst);

                                    [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                                } break;
                            default:
                                {
                                    fprintf(stderr, "%s: node %3d, op = %8s not implemented\n", __func__, i, ggml_op_name(dst->op));
                                    GGML_ASSERT(false);
                                }
                        } break;
                    case GGML_OP_SOFT_MAX:
                        {
                            if (encoder == nil) {
                                encoder = [command_buffer computeCommandEncoderWithDescriptor: edesc];
                            }

                            const int nth = 32;

                            [encoder setComputePipelineState:ctx->pipeline_soft_max];
                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                            [encoder setBytes:&ne00 length:sizeof(ne00) atIndex:2];
                            [encoder setBytes:&ne01 length:sizeof(ne01) atIndex:3];
                            [encoder setBytes:&ne02 length:sizeof(ne02) atIndex:4];
                            [encoder setThreadgroupMemoryLength:nth*sizeof(float) atIndex:0];

                            [encoder dispatchThreadgroups:MTLSizeMake(ne01, ne02, ne03) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
                        } break;
                    case GGML_OP_DIAG_MASK_INF:
                        {
                            if (encoder == nil) {
                                encoder = [command_buffer computeCommandEncoderWithDescriptor: edesc];
                            }

                            const int n_past = ((int32_t *)(dst->op_params))[0];

                            [encoder setComputePipelineState:ctx->pipeline_diag_mask_inf];
                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                            [encoder setBytes:&ne00   length:sizeof(ne00) atIndex:2];
                            [encoder setBytes:&ne01   length:sizeof(ne01) atIndex:3];
                            [encoder setBytes:&n_past length:sizeof(int)  atIndex:4];

                            [encoder dispatchThreadgroups:MTLSizeMake(ne00, ne01, ne02) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                        } break;
                    case GGML_OP_MUL_MAT:
                        {
                            // TODO: needs to be updated after PR: https://github.com/ggerganov/ggml/pull/224

                            GGML_ASSERT(ne00 == ne10);
                            // GGML_ASSERT(ne02 == ne12); // Should be checked on individual data types until broadcast is implemented everywhere
                            GGML_ASSERT(ne03 == ne13);

                            if (ggml_is_contiguous(src0) &&
                                ggml_is_contiguous(src1) &&
                                (src0t == GGML_TYPE_F32 || src0t == GGML_TYPE_F16) && ne11 > 1) {

                                if (encoder != nil) {
                                    [encoder endEncoding];
                                    encoder = nil;
                                }

                                MPSDataType src0dt = src0t == GGML_TYPE_F32 ? MPSDataTypeFloat32 : MPSDataTypeFloat16;
                                MPSDataType src1dt = src1t == GGML_TYPE_F32 ? MPSDataTypeFloat32 : MPSDataTypeFloat16;

                                // for F32 x F32 we use MPS
                                MPSMatrixDescriptor * desc0 = [MPSMatrixDescriptor
                                    matrixDescriptorWithRows:ne01 columns:ne00 rowBytes:src0->nb[1] dataType:src0dt];

                                MPSMatrixDescriptor * desc1 = [MPSMatrixDescriptor
                                    matrixDescriptorWithRows:ne11 columns:ne10 rowBytes:src1->nb[1] dataType:src1dt];

                                MPSMatrixDescriptor * desc  = [MPSMatrixDescriptor
                                    matrixDescriptorWithRows:ne1 columns:ne0 rowBytes:dst->nb[1] dataType:MPSDataTypeFloat32];

                                MPSMatrixMultiplication * mul = [[MPSMatrixMultiplication alloc]
                                    initWithDevice:ctx->device transposeLeft:false transposeRight:true
                                        resultRows:ne11 resultColumns:ne01 interiorColumns:ne00 alpha:1.0 beta:0.0];

                                // we need to do ne12 multiplications
                                // TODO: is there a way to do this in parallel - currently very slow ..
                                // TODO: might be possible to offload part of the computation to ANE using Accelerate's CBLAS
                                for (int64_t i02 = 0; i02 < ne12; ++i02) {
                                    size_t offs_src0_cur = offs_src0 + i02/(ne12/ne02)*nb02; // gqa not used for now
                                    size_t offs_src1_cur = offs_src1 + i02*nb12;
                                    size_t offs_dst_cur  = offs_dst  + i02*nb2;

                                    MPSMatrix * mat_src0 = [[MPSMatrix alloc] initWithBuffer:id_src0 offset:offs_src0_cur descriptor:desc0];
                                    MPSMatrix * mat_src1 = [[MPSMatrix alloc] initWithBuffer:id_src1 offset:offs_src1_cur descriptor:desc1];
                                    MPSMatrix * mat_dst  = [[MPSMatrix alloc] initWithBuffer:id_dst  offset:offs_dst_cur  descriptor:desc ];

                                    [mul encodeToCommandBuffer:command_buffer leftMatrix:mat_src1 rightMatrix:mat_src0 resultMatrix:mat_dst];
                                }
                            } else {
                                if (encoder == nil) {
                                    encoder = [command_buffer computeCommandEncoderWithDescriptor: edesc];
                                }

                                int nth0 = 32;
                                int nth1 = 1;

                                // use custom matrix x vector kernel
                                switch (src0t) {
                                    case GGML_TYPE_F16:
                                        {
                                            nth0 = 64;
                                            nth1 = 1;
                                            [encoder setComputePipelineState:ctx->pipeline_mul_mat_f16_f32];
                                        } break;
                                    case GGML_TYPE_Q4_0:
                                        {
                                            GGML_ASSERT(ne02 == 1);
                                            GGML_ASSERT(ne12 == 1);

                                            nth0 = 8;
                                            nth1 = 8;
                                            [encoder setComputePipelineState:ctx->pipeline_mul_mat_q4_0_f32];
                                        } break;
                                    case GGML_TYPE_Q4_1:
                                        {
                                            GGML_ASSERT(ne02 == 1);
                                            GGML_ASSERT(ne12 == 1);

                                            nth0 = 8;
                                            nth1 = 8;
                                            [encoder setComputePipelineState:ctx->pipeline_mul_mat_q4_1_f32];
                                        } break;
                                    case GGML_TYPE_Q2_K:
                                        {
                                            GGML_ASSERT(ne02 == 1);
                                            GGML_ASSERT(ne12 == 1);

                                            nth0 = 2;
                                            nth1 = 32;
                                            [encoder setComputePipelineState:ctx->pipeline_mul_mat_q2_K_f32];
                                        } break;
                                    case GGML_TYPE_Q3_K:
                                        {
                                            GGML_ASSERT(ne02 == 1);
                                            GGML_ASSERT(ne12 == 1);

                                            nth0 = 2;
                                            nth1 = 32;
                                            [encoder setComputePipelineState:ctx->pipeline_mul_mat_q3_K_f32];
                                        } break;
                                    case GGML_TYPE_Q4_K:
                                        {
                                            GGML_ASSERT(ne02 == 1);
                                            GGML_ASSERT(ne12 == 1);

                                            nth0 = 2;
                                            nth1 = 32;
                                            [encoder setComputePipelineState:ctx->pipeline_mul_mat_q4_K_f32];
                                        } break;
                                    case GGML_TYPE_Q5_K:
                                        {
                                            GGML_ASSERT(ne02 == 1);
                                            GGML_ASSERT(ne12 == 1);

                                            nth0 = 2;
                                            nth1 = 32;
                                            [encoder setComputePipelineState:ctx->pipeline_mul_mat_q5_K_f32];
                                        } break;
                                    case GGML_TYPE_Q6_K:
                                        {
                                            GGML_ASSERT(ne02 == 1);
                                            GGML_ASSERT(ne12 == 1);

                                            nth0 = 2;
                                            nth1 = 32;
                                            [encoder setComputePipelineState:ctx->pipeline_mul_mat_q6_K_f32];
                                        } break;
                                    default:
                                        {
                                            fprintf(stderr, "Asserting on type %d\n",(int)src0t);
                                            GGML_ASSERT(false && "not implemented");
                                        }
                                };

                                [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                                [encoder setBuffer:id_src1 offset:offs_src1 atIndex:1];
                                [encoder setBuffer:id_dst  offset:offs_dst  atIndex:2];
                                [encoder setBytes:&ne00 length:sizeof(ne00) atIndex:3];
                                [encoder setBytes:&ne01 length:sizeof(ne01) atIndex:4];
                                [encoder setBytes:&ne02 length:sizeof(ne02) atIndex:5];
                                [encoder setBytes:&nb00 length:sizeof(nb00) atIndex:6];
                                [encoder setBytes:&nb01 length:sizeof(nb01) atIndex:7];
                                [encoder setBytes:&nb02 length:sizeof(nb02) atIndex:8];
                                [encoder setBytes:&ne10 length:sizeof(ne10) atIndex:9];
                                [encoder setBytes:&ne11 length:sizeof(ne11) atIndex:10];
                                [encoder setBytes:&ne12 length:sizeof(ne12) atIndex:11];
                                [encoder setBytes:&nb10 length:sizeof(nb10) atIndex:12];
                                [encoder setBytes:&nb11 length:sizeof(nb11) atIndex:13];
                                [encoder setBytes:&nb12 length:sizeof(nb12) atIndex:14];
                                [encoder setBytes:&ne0  length:sizeof(ne0)  atIndex:15];
                                [encoder setBytes:&ne1  length:sizeof(ne1)  atIndex:16];

                                if (src0t == GGML_TYPE_Q4_0 || src0t == GGML_TYPE_Q4_1 ||
                                    src0t == GGML_TYPE_Q2_K || src0t == GGML_TYPE_Q4_K) {
                                    [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 7) / 8, ne11, 1) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                                }
                                else if (src0t == GGML_TYPE_Q3_K) {
#ifdef GGML_QKK_64
                                    [encoder dispatchThreadgroups:MTLSizeMake((ne01+1)/2, ne11, 1) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
#else
                                    [encoder dispatchThreadgroups:MTLSizeMake((ne01+3)/4, ne11, 1) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
#endif
                                }
                                else if (src0t == GGML_TYPE_Q5_K) {
                                    [encoder dispatchThreadgroups:MTLSizeMake((ne01 + 3) / 4, ne11, 1) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                                }
                                else if (src0t == GGML_TYPE_Q6_K) {
                                    [encoder dispatchThreadgroups:MTLSizeMake((ne01+1)/2, ne11, 1) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                                } else {
                                    [encoder setThreadgroupMemoryLength:nth0*sizeof(float) atIndex:0];
                                    [encoder dispatchThreadgroups:MTLSizeMake(ne01, ne11, ne12) threadsPerThreadgroup:MTLSizeMake(nth0, nth1, 1)];
                                }
                            }
                        } break;
                    case GGML_OP_GET_ROWS:
                        {
                            if (encoder == nil) {
                                encoder = [command_buffer computeCommandEncoderWithDescriptor: edesc];
                            }

                            switch (src0->type) {
                                case GGML_TYPE_F16:  [encoder setComputePipelineState:ctx->pipeline_get_rows_f16]; break;
                                case GGML_TYPE_Q4_0: [encoder setComputePipelineState:ctx->pipeline_get_rows_q4_0]; break;
                                case GGML_TYPE_Q4_1: [encoder setComputePipelineState:ctx->pipeline_get_rows_q4_1]; break;
                                case GGML_TYPE_Q2_K: [encoder setComputePipelineState:ctx->pipeline_get_rows_q2_K]; break;
                                case GGML_TYPE_Q3_K: [encoder setComputePipelineState:ctx->pipeline_get_rows_q3_K]; break;
                                case GGML_TYPE_Q4_K: [encoder setComputePipelineState:ctx->pipeline_get_rows_q4_K]; break;
                                case GGML_TYPE_Q5_K: [encoder setComputePipelineState:ctx->pipeline_get_rows_q5_K]; break;
                                case GGML_TYPE_Q6_K: [encoder setComputePipelineState:ctx->pipeline_get_rows_q6_K]; break;
                                default: GGML_ASSERT(false && "not implemented");
                            }

                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_src1 offset:offs_src1 atIndex:1];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:2];
                            [encoder setBytes:&(src0->ne[0]) length:sizeof( int64_t) atIndex:3];
                            [encoder setBytes:&(src0->nb[1]) length:sizeof(uint64_t) atIndex:4];
                            [encoder setBytes:&(dst->nb[1])  length:sizeof(uint64_t) atIndex:5];

                            const int64_t n = ggml_nelements(src1);

                            [encoder dispatchThreadgroups:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                        } break;
                    case GGML_OP_RMS_NORM:
                        {
                            if (encoder == nil) {
                                encoder = [command_buffer computeCommandEncoderWithDescriptor: edesc];
                            }

                            float eps;
                            memcpy(&eps, dst->op_params, sizeof(float));

                            const int nth = 512;

                            [encoder setComputePipelineState:ctx->pipeline_rms_norm];
                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                            [encoder setBytes:&ne00 length:sizeof( int64_t) atIndex:2];
                            [encoder setBytes:&nb01 length:sizeof(uint64_t) atIndex:3];
                            [encoder setBytes:&eps  length:sizeof(   float) atIndex:4];
                            [encoder setThreadgroupMemoryLength:nth/32*sizeof(float) atIndex:0];

                            const int64_t nrows = ggml_nrows(src0);

                            [encoder dispatchThreadgroups:MTLSizeMake(nrows, 1, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
                        } break;
                    case GGML_OP_NORM:
                        {
                            if (encoder == nil) {
                                encoder = [command_buffer computeCommandEncoderWithDescriptor: edesc];
                            }

                            const float eps = 1e-5f;

                            const int nth = 256;

                            [encoder setComputePipelineState:ctx->pipeline_norm];
                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                            [encoder setBytes:&ne00 length:sizeof( int64_t) atIndex:2];
                            [encoder setBytes:&nb01 length:sizeof(uint64_t) atIndex:3];
                            [encoder setBytes:&eps  length:sizeof(   float) atIndex:4];
                            [encoder setThreadgroupMemoryLength:nth*sizeof(float) atIndex:0];

                            const int64_t nrows = ggml_nrows(src0);

                            [encoder dispatchThreadgroups:MTLSizeMake(nrows, 1, 1) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
                        } break;
                    case GGML_OP_ALIBI:
                        {
                            if (encoder == nil) {
                                encoder = [command_buffer computeCommandEncoderWithDescriptor: edesc];
                            }

                            GGML_ASSERT((src0t == GGML_TYPE_F32));

                            const int n_past = ((int32_t *) dst->op_params)[0]; UNUSED(n_past);
                            const int n_head = ((int32_t *) dst->op_params)[1];
                            float max_bias;
                            memcpy(&max_bias, (int32_t *) dst->op_params + 2, sizeof(float));

                            if (__builtin_popcount(n_head) != 1) {
                                GGML_ASSERT(false && "only power-of-two n_head implemented");
                            }

                            const int n_heads_log2_floor = 1 << (int) floor(log2(n_head));
                            const float m0 = powf(2.0f, -(max_bias) / n_heads_log2_floor);

                            [encoder setComputePipelineState:ctx->pipeline_alibi_f32];
                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                            [encoder setBytes:&ne00 length:sizeof( int64_t) atIndex:2];
                            [encoder setBytes:&ne01 length:sizeof( int64_t) atIndex:3];
                            [encoder setBytes:&ne02 length:sizeof( int64_t) atIndex:4];
                            [encoder setBytes:&ne03 length:sizeof( int64_t) atIndex:5];
                            [encoder setBytes:&nb00 length:sizeof(uint64_t) atIndex:6];
                            [encoder setBytes:&nb01 length:sizeof(uint64_t) atIndex:7];
                            [encoder setBytes:&nb02 length:sizeof(uint64_t) atIndex:8];
                            [encoder setBytes:&nb03 length:sizeof(uint64_t) atIndex:9];
                            [encoder setBytes:&ne0  length:sizeof( int64_t) atIndex:10];
                            [encoder setBytes:&ne1  length:sizeof( int64_t) atIndex:11];
                            [encoder setBytes:&ne2  length:sizeof( int64_t) atIndex:12];
                            [encoder setBytes:&ne3  length:sizeof( int64_t) atIndex:13];
                            [encoder setBytes:&nb0  length:sizeof(uint64_t) atIndex:14];
                            [encoder setBytes:&nb1  length:sizeof(uint64_t) atIndex:15];
                            [encoder setBytes:&nb2  length:sizeof(uint64_t) atIndex:16];
                            [encoder setBytes:&nb3  length:sizeof(uint64_t) atIndex:17];
                            [encoder setBytes:&m0  length:sizeof(    float) atIndex:18];
                            const int nth = 32;
                            [encoder dispatchThreadgroups:MTLSizeMake(ne01, ne02, ne03) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
                        } break;
                    case GGML_OP_ROPE:
                        {
                            if (encoder == nil) {
                                encoder = [command_buffer computeCommandEncoderWithDescriptor: edesc];
                            }

                            const int n_past = ((int32_t *) dst->op_params)[0];
                            const int n_dims = ((int32_t *) dst->op_params)[1];
                            const int mode   = ((int32_t *) dst->op_params)[2];

                            float freq_base;
                            float freq_scale;
                            memcpy(&freq_base,  (int32_t *) dst->op_params + 4, sizeof(float));
                            memcpy(&freq_scale, (int32_t *) dst->op_params + 5, sizeof(float));

                            [encoder setComputePipelineState:ctx->pipeline_rope];
                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                            [encoder setBytes:&ne00    length:sizeof( int64_t) atIndex:2];
                            [encoder setBytes:&ne01    length:sizeof( int64_t) atIndex:3];
                            [encoder setBytes:&ne02    length:sizeof( int64_t) atIndex:4];
                            [encoder setBytes:&ne03    length:sizeof( int64_t) atIndex:5];
                            [encoder setBytes:&nb00    length:sizeof(uint64_t) atIndex:6];
                            [encoder setBytes:&nb01    length:sizeof(uint64_t) atIndex:7];
                            [encoder setBytes:&nb02    length:sizeof(uint64_t) atIndex:8];
                            [encoder setBytes:&nb03    length:sizeof(uint64_t) atIndex:9];
                            [encoder setBytes:&ne0     length:sizeof( int64_t) atIndex:10];
                            [encoder setBytes:&ne1     length:sizeof( int64_t) atIndex:11];
                            [encoder setBytes:&ne2     length:sizeof( int64_t) atIndex:12];
                            [encoder setBytes:&ne3     length:sizeof( int64_t) atIndex:13];
                            [encoder setBytes:&nb0     length:sizeof(uint64_t) atIndex:14];
                            [encoder setBytes:&nb1     length:sizeof(uint64_t) atIndex:15];
                            [encoder setBytes:&nb2     length:sizeof(uint64_t) atIndex:16];
                            [encoder setBytes:&nb3     length:sizeof(uint64_t) atIndex:17];
                            [encoder setBytes:&n_past  length:sizeof(     int) atIndex:18];
                            [encoder setBytes:&n_dims  length:sizeof(     int) atIndex:19];
                            [encoder setBytes:&mode    length:sizeof(     int) atIndex:20];
                            [encoder setBytes:&freq_base  length:sizeof(float) atIndex:21];
                            [encoder setBytes:&freq_scale length:sizeof(float) atIndex:22];

                            [encoder dispatchThreadgroups:MTLSizeMake(ne01, ne02, ne03) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                        } break;
                    case GGML_OP_DUP:
                    case GGML_OP_CPY:
                    case GGML_OP_CONT:
                        {
                            if (encoder == nil) {
                                encoder = [command_buffer computeCommandEncoderWithDescriptor: edesc];
                            }

                            const int nth = 32;

                            switch (src0t) {
                                case GGML_TYPE_F32:
                                    {
                                        switch (dstt) {
                                            case GGML_TYPE_F16: [encoder setComputePipelineState:ctx->pipeline_cpy_f32_f16]; break;
                                            case GGML_TYPE_F32: [encoder setComputePipelineState:ctx->pipeline_cpy_f32_f32]; break;
                                            default: GGML_ASSERT(false && "not implemented");
                                        };
                                    } break;
                                case GGML_TYPE_F16:
                                    {
                                        switch (dstt) {
                                            case GGML_TYPE_F16: [encoder setComputePipelineState:ctx->pipeline_cpy_f16_f16]; break;
                                            case GGML_TYPE_F32: GGML_ASSERT(false && "cpy_f16_f32 not implemented"); break;
                                            default: GGML_ASSERT(false && "not implemented");
                                        };
                                    } break;
                                default: GGML_ASSERT(false && "not implemented");
                            }

                            [encoder setBuffer:id_src0 offset:offs_src0 atIndex:0];
                            [encoder setBuffer:id_dst  offset:offs_dst  atIndex:1];
                            [encoder setBytes:&ne00 length:sizeof( int64_t) atIndex:2];
                            [encoder setBytes:&ne01 length:sizeof( int64_t) atIndex:3];
                            [encoder setBytes:&ne02 length:sizeof( int64_t) atIndex:4];
                            [encoder setBytes:&ne03 length:sizeof( int64_t) atIndex:5];
                            [encoder setBytes:&nb00 length:sizeof(uint64_t) atIndex:6];
                            [encoder setBytes:&nb01 length:sizeof(uint64_t) atIndex:7];
                            [encoder setBytes:&nb02 length:sizeof(uint64_t) atIndex:8];
                            [encoder setBytes:&nb03 length:sizeof(uint64_t) atIndex:9];
                            [encoder setBytes:&ne0  length:sizeof( int64_t) atIndex:10];
                            [encoder setBytes:&ne1  length:sizeof( int64_t) atIndex:11];
                            [encoder setBytes:&ne2  length:sizeof( int64_t) atIndex:12];
                            [encoder setBytes:&ne3  length:sizeof( int64_t) atIndex:13];
                            [encoder setBytes:&nb0  length:sizeof(uint64_t) atIndex:14];
                            [encoder setBytes:&nb1  length:sizeof(uint64_t) atIndex:15];
                            [encoder setBytes:&nb2  length:sizeof(uint64_t) atIndex:16];
                            [encoder setBytes:&nb3  length:sizeof(uint64_t) atIndex:17];

                            [encoder dispatchThreadgroups:MTLSizeMake(ne01, ne02, ne03) threadsPerThreadgroup:MTLSizeMake(nth, 1, 1)];
                        } break;
                    default:
                        {
                            fprintf(stderr, "%s: node %3d, op = %8s not implemented\n", __func__, i, ggml_op_name(dst->op));
                            GGML_ASSERT(false);
                        }
                }
            }

            if (encoder != nil) {
                [encoder endEncoding];
                encoder = nil;
            }

            [command_buffer commit];
        });
    }

    // wait for all threads to finish
    dispatch_barrier_sync(queue, ^{});

    [command_buffers[n_cb - 1] waitUntilCompleted];

    // check status of command buffers
    // needed to detect if the device ran out-of-memory for example (#1881)
    for (int i = 0; i < n_cb; i++) {
        MTLCommandBufferStatus status = (MTLCommandBufferStatus) [command_buffers[i] status];
        if (status != MTLCommandBufferStatusCompleted) {
            fprintf(stderr, "%s: command buffer %d failed with status %lu\n", __func__, i, status);
            GGML_ASSERT(false);
        }
    }
}
