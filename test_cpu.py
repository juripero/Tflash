from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
import torch.nn.functional as F

from specialized_decode.mono.ffn import reference_ffn, scratch_bytes, scratch_layout
from specialized_decode.mono.residency import checked_residency
from specialized_decode.mono.runner import FfnRunner
from specialized_decode.runtime.executor import FfnExecutor
from specialized_decode.runtime.guards import decide_step, metadata_is_decode_only
from specialized_decode.runtime.spec import DecodeConfig
from specialized_decode.runtime.weights import LayerWeights, bind_model_weights
from specialized_decode.integration import run_decoder_layers
from install_vllm import INIT_OLD, LOOP_OLD, apply_changes, proposed_files


class GemmaRMSNorm(nn.Module):
    def __init__(self, h, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(h, dtype=torch.bfloat16) * .1, requires_grad=False)
        self.variance_epsilon = eps

    def forward(self, hidden_states, residual=None):
        x = hidden_states.float()
        if residual is not None:
            x = x + residual.float()
        resout = x.bfloat16()
        norm = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.variance_epsilon)
        norm = (norm * (1 + self.weight.float())).bfloat16()
        return norm if residual is None else (norm, resout)


class SiluAndMul(nn.Module):
    def forward(self, x):
        gate, up = x.chunk(2, -1)
        return (F.silu(gate.float()) * up.float()).bfloat16()


class MLP(nn.Module):
    def __init__(self, h, inter):
        super().__init__()
        self.gate_up_proj = nn.Linear(h, inter * 2, bias=False, dtype=torch.bfloat16)
        self.down_proj = nn.Linear(inter, h, bias=False, dtype=torch.bfloat16)
        self.act_fn = SiluAndMul()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class LinearAttention(nn.Module):
    # Intentionally rejects `positions` and arbitrary kwargs.
    def forward(self, hidden_states):
        return hidden_states * .125


class FullAttention(nn.Module):
    def forward(self, hidden_states, positions):
        if positions.dtype != torch.int64 or positions.ndim != 1:
            raise AssertionError("positions were misrouted")
        return hidden_states * .25


class Layer(nn.Module):
    def __init__(self, kind, config, scale=False):
        super().__init__()
        self.layer_type = kind
        self.layer_scale = scale
        h = config.hidden_size
        self.input_layernorm = GemmaRMSNorm(h, config.eps)
        self.post_attention_layernorm = GemmaRMSNorm(h, config.eps)
        self.mlp = MLP(h, config.intermediate_size)
        if kind == "linear_attention":
            self.linear_attn = LinearAttention()
        else:
            self.self_attn = FullAttention()
        if scale:
            self.attn_layer_scale = nn.Parameter(torch.randn(1, 1, h) * .1, requires_grad=False)
            self.ffn_layer_scale = nn.Parameter(torch.randn(1, 1, h) * .1, requires_grad=False)

    # Match upstream argument order exactly, to expose positional-call bugs.
    def forward(self, hidden_states, residual, positions):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states=hidden_states)
        else:
            hidden_states = self.self_attn(hidden_states=hidden_states, positions=positions)
        if self.layer_scale:
            hidden_states = hidden_states * (self.attn_layer_scale.to(hidden_states.dtype)[0] + 1)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        if self.layer_scale:
            hidden_states = hidden_states * (self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1)
        return hidden_states, residual


class CpuTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(18)
        self.config = DecodeConfig(hidden_size=32, intermediate_size=64, num_layers=2)

    def executor(self, scale=False):
        layers = nn.ModuleList([Layer("linear_attention", self.config, scale),
                                Layer("full_attention", self.config, scale)])
        model = SimpleNamespace(layers=layers)
        weights = bind_model_weights(model, self.config, require_cuda=False)
        runner = FfnRunner(torch.device("cpu"), self.config)
        runner.warmup()
        return FfnExecutor(layers, weights, runner)

    def test_active_and_stock_layer_parity_including_residual_and_scaling(self):
        with torch.no_grad():
            for scale in (False, True):
                executor = self.executor(scale)
                for rows in (1, 3, 16):
                    for first in (False, True):
                        positions = torch.arange(rows)
                        h = torch.randn(rows, 32).bfloat16()
                        r = None if first else torch.randn_like(h)
                        for i in (0, 1):
                            expected = executor.layers[i](h, r, positions)
                            for active in (False, True):
                                actual = executor.run_layer(i, positions, h, r, active=active)
                                for a, b in zip(actual, expected):
                                    torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_bound_weights_are_zero_copy_and_replacement_is_rejected(self):
        executor = self.executor()
        self.assertIs(executor.weights[0].w13, executor.layers[0].mlp.gate_up_proj.weight)
        executor.layers[0].mlp.down_proj.weight = nn.Parameter(executor.weights[0].w2.clone())
        with self.assertRaisesRegex(RuntimeError, "replaced"):
            executor.run_layer(0, torch.arange(1), torch.ones(1, 32).bfloat16(), None, active=True)

    def test_scratch_layout_returns_sizes_and_nonoverlapping_regions(self):
        for rows in range(1, 17):
            layout = scratch_layout(rows, 32, 64)
            self.assertEqual(layout["act_mid"][0], layout["x_norm"][1])
            self.assertEqual(scratch_bytes(rows, 32, 64), rows * (32 + 64) * 2)
            self.assertIsInstance(scratch_bytes(rows), int)

    def test_model_loop_preserves_aux_states_and_checks_stock_steps(self):
        executor = self.executor(scale=True)
        positions = torch.arange(3)
        initial = torch.randn(3, 32).bfloat16()
        expected, res_expected = initial, None
        with torch.no_grad():
            for layer in executor.layers:
                expected, res_expected = layer(expected, res_expected, positions)
        for active in (False, True):
            calls = []
            inner = SimpleNamespace(
                layers=executor.layers, start_layer=0, end_layer=2,
                _maybe_add_hidden_state=lambda aux, idx, h, r: aux.append(idx))
            controller = SimpleNamespace(
                executor=executor,
                plan=lambda *args: SimpleNamespace(use_mono=active),
                after_step=lambda error: calls.append(error))
            aux = []
            with patch("specialized_decode.integration.get_or_build_decode", return_value=controller):
                out, res = run_decoder_layers(inner, positions, initial, None, None, aux)
            torch.testing.assert_close(out, expected, atol=0, rtol=0)
            torch.testing.assert_close(res, res_expected, atol=0, rtol=0)
            self.assertEqual(aux, [1, 2])
            self.assertEqual(calls, [None])

    def test_decode_decision_fails_closed_and_keeps_physical_rows(self):
        for rows in range(1, 17):
            self.assertEqual(decide_step(rows, is_decode_only=True).rows, rows)
            self.assertFalse(decide_step(rows).use_mono)
        for rows in (0, 17):
            self.assertFalse(decide_step(rows, is_decode_only=True).use_mono)
        self.assertFalse(decide_step(1, is_decode_only=True, is_piecewise_cudagraph=True).use_mono)
        self.assertFalse(metadata_is_decode_only({"a": SimpleNamespace(max_query_len=1)}))
        self.assertTrue(metadata_is_decode_only({"gdn": SimpleNamespace(num_prefills=0, num_decodes=3)}))
        self.assertFalse(metadata_is_decode_only({"gdn": SimpleNamespace(num_prefills=1, num_decodes=3)}))

    def test_invalid_config_and_unknown_residency_are_rejected(self):
        for args in ({"max_rows": 17}, {"eps": float("nan")}, {"blocks": 0}, {"backend": "flydsl"}):
            with self.assertRaises(ValueError):
                DecodeConfig(**args)
        for values in ([0, 64, 32], [1, 32, 32], [2, 64, 64]):
            with self.assertRaises(RuntimeError):
                checked_residency(SimpleNamespace(occupancy=lambda: values), 64)
        self.assertEqual(checked_residency(SimpleNamespace(occupancy=lambda: [2, 32, 32]), 64).capacity, 64)

    def test_health_uses_scalar_and_failure_stays_sticky(self):
        runner = self.executor().runner
        runner.check_health()
        runner.abort()
        for _ in range(2):
            with self.assertRaises(RuntimeError):
                runner.check_health()
        self.assertEqual(runner.health()[0], 1)

    def test_noncontiguous_and_excess_rows_are_rejected(self):
        executor = self.executor()
        for a in (torch.empty(32, 2, dtype=torch.bfloat16).T,
                  torch.empty(17, 32, dtype=torch.bfloat16)):
            with self.assertRaises(ValueError):
                executor.runner.launch_layer(executor.weights[0], a, a)

    def test_installer_validates_both_files_before_writing_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "vllm/model_executor/models"
            base.mkdir(parents=True)
            p = base / "qwen3_5.py"
            q = base / "qwen3_next.py"
            p.write_text("class Model:\n    def init(self):\n" + INIT_OLD)
            q.write_text("class Model:\n    def forward(self):\n" + LOOP_OLD)
            changes = proposed_files(root)
            apply_changes(changes)
            self.assertTrue(p.with_suffix(".py.qwen38.bak").exists())
            self.assertTrue(all(before == after for _, before, after in proposed_files(root)))
            q.write_text("unrelated = 1\n")
            before = p.read_text()
            with self.assertRaises(RuntimeError):
                proposed_files(root)
            self.assertEqual(p.read_text(), before)


if __name__ == "__main__":
    unittest.main()
