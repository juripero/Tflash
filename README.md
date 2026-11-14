# Qwen 27B TP1 BF16 decode corrections

This package replaces the reviewed `juripero/Temp/specialized_decode` snapshot
`dfebe6d884eeb6912a6fca247755fc257f6097cc`. Integration was checked against vLLM
`d64414d044a898c6db646dcecd122213accd1732`.

The requested “Qwen3.8” model is treated as the Qwen3.5-compatible architecture
specified by the original code: `qwen3_5_text`, 64 layers, hidden size 5120,
intermediate size 17408, dense SwiGLU, Gemma-style `1 + weight` RMSNorm, epsilon
1e-6. The runtime validates these values instead of assuming another checkpoint
with a similar name has this layout. TP1, PP1, unquantized BF16, no LoRA, and no
speculative decoding are required. Weight capacity is assumed sufficient; there
is no arbitrary 64-GiB capacity check.

## What is delivered

- `backend="torch"` (default): staged BF16 reference implementation, intended
  for correctness and integration checks. It is not a claimed acceleration.
- `backend="hip"`: one persistent HIP/WMMA FFN launch per layer on gfx1201.
  HIP source is embedded in `mono/a16w16.py` and compiled during explicit warmup
  through PyTorch's extension loader. This is a replacement for the incompatible
  FlyDSL implementation, not a FlyDSL patch. It requires a working ROCm compiler,
  ROCm PyTorch, a C++ compiler, and Ninja in the serving environment.
- `install_vllm.py`: source-shape-checked integration edits, dry-run by default.
- `tests/test_cpu.py`: executable host and staged-math regression tests.
- `validate.py`: target-GPU stage parity, width coverage, graph replay, generation
  wrap, and dropped-CTA failure tests.

All files in the original module layout are replaced. `runtime/__init__.py` and
`integration.py` are added. The public controller/runner APIs have changed;
remove old monkey-patching integration rather than combining both approaches.

## Validation status

CPU tests and Python compilation passed in the creation environment. The
integration patch was dry-run against the pinned vLLM files. There was no ROCm
GPU or HIP compiler in that environment, so the HIP source has **not** been
compiled, run, benchmarked, or qualified on gfx1201. Do not interpret source
corrections as a measured speedup or production qualification.

The vLLM adapter in this release deliberately requires `--enforce-eager`.
Standalone persistent-kernel HIP graph capture is implemented and has a supplied
qualification test. Full vLLM graph/Dynamo/Inductor integration is not claimed;
the static guard refuses it and retains stock execution. Supporting vLLM graphs
requires a dedicated custom-op/model-runner integration and end-to-end testing.

## Install and inspect

Use the same Python environment as your ROCm vLLM installation. Install this
package without resolving or replacing your existing ROCm torch dependency:

```bash
python -m pip install -e . --no-deps
python -m unittest discover -s tests -v
python install_vllm.py /path/to/vllm
```

The last command prints a diff and changes nothing. It validates both target
files before writing. To apply the reviewed integration:

```bash
python install_vllm.py /path/to/vllm --apply
```

The installer preserves `qwen3_5.py.qwen38.bak` and
`qwen3_next.py.qwen38.bak`. Apply to the checkout actually imported by your vLLM
worker, restart workers afterward, and review the diff. A source-anchor mismatch
is a refusal, not an invitation to blindly replace text. Do not apply this over
an existing custom Qwen FFN patch without reconciling the two integrations.

## Qualify the HIP backend

On an otherwise idle gfx1201, start with small shapes, then the actual FFN sizes:

```bash
python validate.py --small --graph --fault-test
python validate.py --graph --fault-test
```

These commands build HIP source and run on the GPU. They exit nonzero on a
compile, occupancy, numerical, synchronization, or graph failure. The fault test
drops one CTA from the first barrier; it inspects the error flag without firing
the production device assertion and discards the failed workspace afterward.

Default numerical thresholds are `atol=0.03125`, `rtol=0.02`; they are starting
qualification criteria, not guaranteed model-quality bounds. Residual outputs
are checked exactly. Norm, SwiGLU, and down output errors are printed separately.
Do not loosen tolerances merely to make an unexplained mismatch pass. Compare
loaded-model per-layer outputs, final logits, deterministic generations, and
evaluation quality as well. Different GEMM reduction orders can cause BF16
rounding differences even when index maps are correct.

## Use with vLLM

First exercise the staged integration with your actual checkpoint:

```bash
QWEN38_SPECIALIZED_DECODE=1 QWEN38_FFN_BACKEND=torch \
  vllm serve /path/to/checkpoint --dtype bfloat16 \
  --tensor-parallel-size 1 --enforce-eager
```

After GPU qualification and model-output comparison, select the HIP backend:

```bash
QWEN38_SPECIALIZED_DECODE=1 QWEN38_FFN_BACKEND=hip \
  vllm serve /path/to/checkpoint --dtype bfloat16 \
  --tensor-parallel-size 1 --enforce-eager
```

Optionally set `QWEN38_FFN_CACHE_DIR` to a writable compiler-cache directory.
With `QWEN38_SPECIALIZED_DECODE` unset or equal to `0`, patched model loops run
stock layers. Unknown/mixed/prefill metadata and physical row counts above 16
also use stock layers. Missing GDN decode metadata fails closed. Inspect startup
logs: configuration refusals explicitly say that stock execution remains active.
Compare performance against a stock run using the same `--enforce-eager` setting.

## Correction map

| Reviewed defect | Replacement |
|---|---|
| Markdown in runner; invalid nesting | Valid Python package; host compilation outside the kernel |
| Tuple/integer scratch arithmetic | Explicit `(offset, size)` indexing; integer byte counts |
| FlyDSL API mismatch | Python-managed HIP extension with explicit stream launch |
| Down fragment append/partial stores | Complete 16-K fragments; stores only after full K reduction |
| Positional fallback mismatch | Keyword calls to original layers |
| DeltaNet given positions | Strict `linear_attn(hidden_states=...)` call |
| Omitted layer scaling | Attention and FFN scales preserved |
| Reused arrival count | Acquire/release reusable generation barrier, device-owned state |
| Host epochs frozen in graphs | No host epoch; device generation advances on every replay |
| WMMA token/feature swap | Token=`lane%16`, feature=`tile*16+(lane//16)*8+i` |
| Four waves duplicating work | Independent feature tiles per wave |
| Single-CTA norm and invalid broadcast | CTA-per-row norm with real LDS scalar and shuffle reduction |
| FP32/BF16 norm inconsistency | FP32 residual sum used for variance and normalized numerator |
| Gate/up rounding mismatch | BF16 gate/up projection boundaries before FP32 SwiGLU |
| Out-of-bounds inactive loads | Actual conditional loads; fixed maximum-row scratch capacity |
| Occupancy assumed when unknown | Query actual compiled kernel; fail on insufficient capacity |
| Vector `_assert_async` | Scalar sticky-failure check after every integrated forward |
| Missing model-loop wiring | Explicit plan, layer dispatch, auxiliary collection, after-step check |
| Captured actual-token count | Launch uses fixed physical rows; padding rows are independent |
| Arbitrary capacity/CU assumptions | No capacity cutoff; residency uses HIP's own processor units |

## Execution contracts

Each runner owns one scratch/control region and binds to one HIP execution
stream. Do not concurrently replay graphs sharing that workspace, even on
different streams. Keep the runner and weights alive for the lifetime of captured
graphs. A graph's physical row shape is fixed; varying the number of valid rows
within that shape is safe because the FFN is per-token. Padding rows may produce
arbitrary outputs, including NaNs, and must remain padding downstream. Stock
attention exclusively owns cache/state updates and padding masks.

The software barrier uses device-scope atomics and a bounded spin count. The
bound is an iteration count, not a ten-second timer. Occupancy establishes
resource capacity, not exclusive scheduling under competing processes; use an
idle/dedicated device for qualification. A fault is sticky and must stop the
worker. There is no public clear-and-resume operation for a failed workspace.

The adapter builds the controller on its first eager forward, after weights
have been loaded. Weight tensors are retained without copies. Replacing weight
storage later is rejected on the active eager path; rebuild the controller and
any graphs after reloads. In-place value updates must also be serialized with GPU
work and revalidated. Neither training nor concurrent model mutation is supported.

## Performance expectations

This is a correctness-oriented persistent implementation. It uses contiguous
BF16 weight fragments and native gfx1201 WMMA, but has no tuned prefetch pipeline,
weight preshuffle, or benchmark-selected launch geometry. It does not promise to
beat hipBLASLt. The declared dense FFNs contain 31.875 GiB of BF16 weights across
64 layers, so bandwidth remains central even when the weights fit in VRAM.

The next optimization work should follow profiling: tune workgroup count and
register pressure, test weight-load vectorization/prefetch, compare small-M
kernels at 1/2/4/8/16 rows, and then validate full-model latency and quality.

## Source references

- Original: https://github.com/juripero/Temp/tree/dfebe6d884eeb6912a6fca247755fc257f6097cc/specialized_decode
- MiniMax reference: https://github.com/vllm-project/vllm/tree/d64414d044a898c6db646dcecd122213accd1732/vllm/models/minimax_m3/amd/specialized_decode
- Qwen model: https://github.com/vllm-project/vllm/blob/d64414d044a898c6db646dcecd122213accd1732/vllm/model_executor/models/qwen3_5.py
- RDNA4 WMMA layout: https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-1/
- HIP occupancy: https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/occupancy.html
