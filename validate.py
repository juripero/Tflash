"""GPU qualification for the submitted HIP source; exits nonzero on failure.

Examples:
  python validate.py --small --graph --fault-test
  python validate.py --graph

Run in a disposable process before enabling QWEN38_FFN_BACKEND=hip.
Random-weight parity is necessary but does not replace loaded-model logit/eval
comparisons. BF16 reduction-order differences are measured, not hidden.
"""
from __future__ import annotations

import argparse
import json
import math

import torch

from specialized_decode.mono.ffn import reference_ffn
from specialized_decode.mono.runner import FfnRunner
from specialized_decode.runtime.spec import DecodeConfig
from specialized_decode.runtime.weights import LayerWeights


def compare(name, actual, expected, atol, rtol):
    error = (actual.float() - expected.float()).abs()
    result = {"stage": name, "max_abs": error.max().item(),
              "rmse": error.square().mean().sqrt().item()}
    print(json.dumps(result), flush=True)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def run(args):
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("GPU qualification requires ROCm PyTorch on gfx1201")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
    if arch != "gfx1201":
        raise RuntimeError(f"Expected gfx1201, found {arch}")
    torch.manual_seed(3107)
    h, inter = (32, 64) if args.small else (5120, 17408)
    config = DecodeConfig(backend="hip", hidden_size=h, intermediate_size=inter,
                          num_layers=1, blocks=args.blocks, cache_dir=args.cache_dir)
    weights = LayerWeights(
        0, (torch.randn(h, device=device) * .1).bfloat16(),
        (torch.randn(inter * 2, h, device=device) / math.sqrt(h)).bfloat16(),
        (torch.randn(h, inter, device=device) / math.sqrt(inter)).bfloat16(), config.eps,
    )
    runner = FfnRunner(device, config)
    runner.warmup()
    print(json.dumps({"residency": vars(runner.residency), "dims": [h, inter]}))
    stream = torch.cuda.Stream(device=device)
    torch.cuda.synchronize(device)
    with torch.cuda.stream(stream), torch.no_grad():
        # Non-power-of-two widths exercise physical-row bounds and masking.
        for rows in (1, 2, 3, 4, 7, 8, 15, 16):
            a = torch.randn(rows, h, device=device).bfloat16()
            r = torch.randn_like(a)
            expected, res_expected, x_expected, mid_expected = reference_ffn(
                a, r, weights, return_stages=True)
            actual, res_actual = runner.launch_layer(weights, a, r)
            stream.synchronize()
            if runner.health()[0]:
                raise RuntimeError(f"Kernel fault: {runner.health()}")
            compare(f"rows={rows}/residual", res_actual, res_expected, 0, 0)
            x_actual = runner.scratch[:rows * h].view(rows, h)
            mid_actual = runner.scratch[16 * h:16 * h + rows * inter].view(rows, inter)
            compare(f"rows={rows}/norm", x_actual, x_expected, args.atol, args.rtol)
            compare(f"rows={rows}/swiglu", mid_actual, mid_expected, args.atol, args.rtol)
            compare(f"rows={rows}/output", actual, expected, args.atol, args.rtol)

        # Reuse across the unsigned generation wrap boundary. Two barriers
        # per launch must take 0xfffffffe to 0 without a special host reset.
        stream.synchronize()
        runner.control[1].fill_(-2)
        a = torch.randn(8, h, device=device).bfloat16()
        r = torch.randn_like(a)
        actual, res_actual = runner.launch_layer(weights, a, r)
        expected, res_expected = reference_ffn(a, r, weights)
        stream.synchronize()
        if runner.control.tolist() != [0, 0]:
            raise AssertionError(f"Counter/generation did not wrap cleanly: {runner.control.tolist()}")
        compare("generation-wrap", actual, expected, args.atol, args.rtol)

        if args.graph:
            # Warm the allocation and scalar-assert paths on this same stream.
            for _ in range(2):
                runner.launch_layer(weights, a, r)
                runner.check_health()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                graph_out, graph_res = runner.launch_layer(weights, a, r)
                runner.check_health()
            for iteration in range(24):
                valid = 1 + iteration % 8
                a.copy_(torch.randn_like(a))
                r.copy_(torch.randn_like(r))
                # Padded rows may contain arbitrary values. Poison them to
                # demonstrate that valid-token columns remain independent.
                a[valid:].fill_(float("nan"))
                r[valid:].fill_(float("nan"))
                expected, res_expected = reference_ffn(a[:valid], r[:valid], weights)
                graph.replay()
                stream.synchronize()
                compare(f"graph/{iteration}/valid={valid}", graph_out[:valid],
                        expected, args.atol, args.rtol)
                compare(f"graph/{iteration}/residual", graph_res[:valid], res_expected, 0, 0)

    if args.fault_test:
        if args.blocks < 2:
            raise ValueError("fault-test requires at least two CTAs")
        # A failed workspace is discarded permanently. Do not reuse the good
        # runner or call its normal assert, which intentionally traps a worker.
        bad = FfnRunner(device, config)
        bad.warmup()
        with torch.cuda.stream(stream):
            a = torch.zeros(1, h, device=device, dtype=torch.bfloat16)
            out, resout = torch.empty_like(a), torch.empty_like(a)
            bad._raw_launch(weights, a, a, out, resout, 1, stream.cuda_stream,
                            drop_cta=0, max_spins=10000)
            stream.synchronize()
            first = bad.health()
            if first[0] != 1 or first[1] != 1:
                raise AssertionError(f"Missing timeout flag: {first}")
            bad._raw_launch(weights, a, a, out, resout, 1, stream.cuda_stream)
            stream.synchronize()
            if bad.health() != first:
                raise AssertionError("Failure was not sticky")
            print(json.dumps({"fault_injection": "passed", "fault": first}))
    print("GPU qualification passed for these shapes and tolerances.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--fault-test", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--blocks", type=int, default=64)
    parser.add_argument("--cache-dir")
    parser.add_argument("--atol", type=float, default=.03125)
    parser.add_argument("--rtol", type=float, default=.02)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
