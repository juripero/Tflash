"""Compile the gfx1201 WMMA kernel from Python through PyTorch's HIP extension.

The original FlyDSL code used incompatible APIs. This implementation has no
FlyDSL dependency. Compilation occurs explicitly during warmup, never capture.
The embedded HIP is supplied as source, not a hardware-qualified binary.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

CPP = r'''
#include <cstdint>
#include <vector>
void ffn_launch(int64_t a, int64_t r, int64_t norm, int64_t w13, int64_t w2,
                int64_t out, int64_t resout, int64_t scratch, int64_t control,
                int64_t fault, int rows, double eps, int blocks,
                int max_spins, int drop_cta, int64_t stream);
std::vector<int64_t> occupancy();
'''

HIP = r'''
#include <hip/hip_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>

constexpr int H = @HIDDEN@;
constexpr int I = @INTERMEDIATE@;
constexpr int MAX_ROWS = 16;
constexpr int THREADS = 128;
constexpr int WAVES = 4;
using u16 = uint16_t;
using u32 = unsigned int;
using v8s = short __attribute__((ext_vector_type(8)));
using v8f = float __attribute__((ext_vector_type(8)));

#define HIP_CHECK(expr) do { hipError_t e = (expr); if (e != hipSuccess) \
  throw std::runtime_error(std::string(#expr) + ": " + hipGetErrorString(e)); } while (0)

__device__ __forceinline__ float as_float(u16 x) {
  return __uint_as_float(static_cast<u32>(x) << 16);
}
__device__ __forceinline__ u16 as_bf16(float x) {
  u32 bits = __float_as_uint(x);
  if ((bits & 0x7fffffffU) > 0x7f800000U)
    return static_cast<u16>((bits >> 16) | 0x0040U);
  return static_cast<u16>((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}
__device__ __forceinline__ u32 acquire(u32* p) {
  return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ void release(u32* p, u32 value) {
  __hip_atomic_store(p, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ void mark_fault(u32* fault, u32 site, u32 gen) {
  if (atomicCAS(fault, 0U, 1U) == 0U) {
    fault[1] = site;
    fault[2] = blockIdx.x;
    fault[3] = gen;
  }
}

__device__ __noinline__ bool grid_barrier(u32* control, u32* fault,
                                         int max_spins, u32 site) {
  __shared__ int ok;
  // All producing threads publish their writes before their CTA leader joins.
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    u32 generation = acquire(control + 1);
    u32 arrived = __hip_atomic_fetch_add(control, 1U, __ATOMIC_ACQ_REL,
                                         __HIP_MEMORY_SCOPE_AGENT);
    if (arrived == gridDim.x - 1) {
      __hip_atomic_store(control, 0U, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      release(control + 1, generation + 1U);
    } else {
      int spins = 0;
      while (acquire(control + 1) == generation && acquire(fault) == 0U) {
        if (++spins >= max_spins) {
          mark_fault(fault, site, generation);
          break;
        }
        #if defined(__HIP_DEVICE_COMPILE__)
        __builtin_amdgcn_s_sleep(1);
        #endif
      }
    }
    ok = acquire(fault) == 0U;
  }
  __syncthreads();
  return ok != 0;
}

__device__ __forceinline__ v8f wmma(v8s a, v8s b, v8f c) {
  #if defined(__HIP_DEVICE_COMPILE__)
  #if !defined(__gfx1201__)
  #error "This kernel is intentionally restricted to gfx1201"
  #endif
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
  #else
  return c;
  #endif
}

template<int K>
__device__ __forceinline__ v8s weight_fragment(const u16* w, int group, int k) {
  int lane = threadIdx.x & 31;
  int row = group * 16 + (lane & 15);
  int first_k = k + (lane >> 4) * 8;
  v8s v;
  #pragma unroll
  for (int j = 0; j < 8; ++j) v[j] = static_cast<short>(w[row * K + first_k + j]);
  return v;
}

template<int K>
__device__ __forceinline__ v8s activation_fragment(const u16* x, int k, int rows) {
  int lane = threadIdx.x & 31;
  int token = lane & 15;
  int first_k = k + (lane >> 4) * 8;
  v8s v = {};
  // Real control-flow predicate: inactive lanes never dereference input.
  if (token < rows) {
    #pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = static_cast<short>(x[token * K + first_k + j]);
  }
  return v;
}

__global__ __launch_bounds__(THREADS) void dense_ffn(
    const u16* a, const u16* r, const u16* norm, const u16* w13,
    const u16* w2, u16* out, u16* resout, u16* scratch,
    u32* control, u32* fault, int rows, float eps, int max_spins, int drop_cta) {
  __shared__ float reduction[WAVES];
  __shared__ float inv_rms;
  __shared__ int alive;
  int tid = threadIdx.x;
  int wave = tid >> 5;
  int lane = tid & 31;
  if (tid == 0) alive = acquire(fault) == 0U;
  __syncthreads();
  if (!alive) return;
  // Fault injection for validate.py only; production passes -1.
  if (static_cast<int>(blockIdx.x) == drop_cta) return;
  u16* x = scratch;
  u16* mid = scratch + MAX_ROWS * H;

  // One CTA per token, not one CTA for all tokens. More than one token per
  // CTA is supported when a deliberately small grid is used for testing.
  for (int token = blockIdx.x; token < rows; token += gridDim.x) {
    float ssq = 0.0f;
    for (int j = tid; j < H; j += THREADS) {
      float sum = as_float(a[token * H + j]) + as_float(r[token * H + j]);
      resout[token * H + j] = as_bf16(sum);
      ssq += sum * sum;
    }
    #pragma unroll
    for (int delta = 16; delta >= 1; delta >>= 1)
      ssq += __shfl_down(ssq, delta, 32);
    if (lane == 0) reduction[wave] = ssq;
    __syncthreads();
    if (tid == 0) {
      float total = 0.0f;
      #pragma unroll
      for (int w = 0; w < WAVES; ++w) total += reduction[w];
      inv_rms = 1.0f / sqrtf(total / H + eps);
    }
    __syncthreads();
    for (int j = tid; j < H; j += THREADS) {
      // Recompute FP32 sum: do not normalize a rounded BF16 residual.
      float sum = as_float(a[token * H + j]) + as_float(r[token * H + j]);
      x[token * H + j] = as_bf16(sum * inv_rms * (1.0f + as_float(norm[j])));
    }
    __syncthreads();
  }
  if (!grid_barrier(control, fault, max_spins, 1)) return;

  // Each wave owns distinct output features; no duplicate wave stores.
  for (int group = blockIdx.x * WAVES + wave; group < I / 16;
       group += gridDim.x * WAVES) {
    v8f gate = {}, up = {};
    for (int k = 0; k < H; k += 16) {
      v8s xv = activation_fragment<H>(x, k, rows);
      gate = wmma(weight_fragment<H>(w13, group, k), xv, gate);
      up = wmma(weight_fragment<H>(w13, group + I / 16, k), xv, up);
    }
    int token = lane & 15;
    if (token < rows) {
      #pragma unroll
      for (int j = 0; j < 8; ++j) {
        int feature = group * 16 + (lane >> 4) * 8 + j;
        // Stock gate_up_proj materializes BF16 before activation.
        float g = as_float(as_bf16(gate[j]));
        float u = as_float(as_bf16(up[j]));
        mid[token * I + feature] = as_bf16((g / (1.0f + expf(-g))) * u);
      }
    }
  }
  if (!grid_barrier(control, fault, max_spins, 2)) return;

  for (int group = blockIdx.x * WAVES + wave; group < H / 16;
       group += gridDim.x * WAVES) {
    v8f acc = {};
    for (int k = 0; k < I; k += 16)
      acc = wmma(weight_fragment<I>(w2, group, k),
                 activation_fragment<I>(mid, k, rows), acc);
    int token = lane & 15;
    if (token < rows) {
      #pragma unroll
      for (int j = 0; j < 8; ++j) {
        int feature = group * 16 + (lane >> 4) * 8 + j;
        out[token * H + feature] = as_bf16(acc[j]);
      }
    }
  }
}

std::vector<int64_t> occupancy() {
  int device = 0, count = 0;
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&props, device));
  std::string arch(props.gcnArchName);
  if (arch.substr(0, arch.find(':')) != "gfx1201")
    throw std::runtime_error("gfx1201 is required");
  HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&count, dense_ffn, THREADS, 0));
  return {count, props.multiProcessorCount, props.warpSize};
}

void ffn_launch(int64_t a, int64_t r, int64_t norm, int64_t w13, int64_t w2,
                int64_t out, int64_t resout, int64_t scratch, int64_t control,
                int64_t fault, int rows, double eps, int blocks,
                int max_spins, int drop_cta, int64_t stream) {
  if (rows < 1 || rows > MAX_ROWS || blocks < 1 || blocks > 64 || max_spins < 1)
    throw std::runtime_error("invalid FFN launch geometry");
  hipLaunchKernelGGL(dense_ffn, dim3(blocks), dim3(THREADS), 0,
      reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const u16*>(a),
      reinterpret_cast<const u16*>(r), reinterpret_cast<const u16*>(norm),
      reinterpret_cast<const u16*>(w13), reinterpret_cast<const u16*>(w2),
      reinterpret_cast<u16*>(out), reinterpret_cast<u16*>(resout),
      reinterpret_cast<u16*>(scratch), reinterpret_cast<u32*>(control),
      reinterpret_cast<u32*>(fault), rows, static_cast<float>(eps), max_spins, drop_cta);
  HIP_CHECK(hipGetLastError());
}
'''


def hip_source(hidden_size: int, intermediate_size: int) -> str:
    if any(n <= 0 or n % 16 for n in (hidden_size, intermediate_size)):
        raise ValueError("dimensions must be positive multiples of 16")
    return HIP.replace("@HIDDEN@", str(hidden_size)).replace("@INTERMEDIATE@", str(intermediate_size))


@lru_cache(maxsize=None)
def build_ffn(hidden_size=5120, intermediate_size=17408, cache_dir=None):
    import torch
    from torch.utils.cpp_extension import load_inline

    if torch.version.hip is None:
        raise RuntimeError("HIP backend requires a ROCm PyTorch build and ROCm compiler")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Compile/warmup must precede HIP graph capture")
    source = hip_source(hidden_size, intermediate_size)
    digest = hashlib.sha256((CPP + source + torch.__version__).encode()).hexdigest()[:12]
    name = f"qwen38_bf16_gfx1201_{digest}"
    directory = None
    if cache_dir is not None:
        directory = str(Path(cache_dir).expanduser().resolve() / name)
        Path(directory).mkdir(parents=True, exist_ok=True)
    return load_inline(
        name=name, cpp_sources=CPP, cuda_sources=source,
        functions=["ffn_launch", "occupancy"],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--offload-arch=gfx1201",
                          "-mwavefrontsize32", "-fno-fast-math"],
        with_cuda=True, build_directory=directory, verbose=False,
    )
