"""Apply the two model-loop edits against the reviewed vLLM source shape.

Default: print a diff. --apply: preserve backups and apply both edits. This
script does not install the Python package; use `pip install -e . --no-deps`.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import os
import shutil
import tempfile
from pathlib import Path

SOURCE_COMMIT = "d64414d044a898c6db646dcecd122213accd1732"
INIT_OLD = "        self.config = config\n        self.quant_config = vllm_config.quant_config\n"
INIT_NEW = (INIT_OLD + "        self._qwen38_vllm_config = vllm_config  # qwen38-decode-adapter-v1\n")
LOOP_OLD = '''        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            self._maybe_add_hidden_state(
                aux_hidden_states, layer_idx + 1, hidden_states, residual
            )
'''
LOOP_NEW = '''        # qwen38-decode-adapter-v1
        if getattr(self, "_qwen38_vllm_config", None) is not None:
            from specialized_decode.integration import run_decoder_layers
            hidden_states, residual = run_decoder_layers(
                self, positions, hidden_states, residual,
                intermediate_tensors, aux_hidden_states,
            )
        else:
''' + "".join("    " + line + "\n" for line in LOOP_OLD.splitlines())


def proposed_files(root: Path):
    base = root / "vllm/model_executor/models"
    changes = []
    for filename, old, new in (("qwen3_5.py", INIT_OLD, INIT_NEW),
                               ("qwen3_next.py", LOOP_OLD, LOOP_NEW)):
        path = base / filename
        text = path.read_text()
        if new in text:
            changes.append((path, text, text))
            continue
        if text.count(old) != 1:
            raise RuntimeError(f"{filename}: source does not match reviewed integration point; no files written")
        updated = text.replace(old, new, 1)
        ast.parse(updated, filename=str(path))
        changes.append((path, text, updated))
    return changes


def apply_changes(changes):
    todo = [(p, before, after) for p, before, after in changes if before != after]
    for path, before, _ in todo:
        if path.read_text() != before:
            raise RuntimeError(f"{path}: changed since inspection")
        if path.with_suffix(path.suffix + ".qwen38.bak").exists():
            raise RuntimeError(f"{path}: backup already exists; review before retrying")
    written = []
    try:
        for path, _, after in todo:
            backup = path.with_suffix(path.suffix + ".qwen38.bak")
            shutil.copy2(path, backup)
            written.append((path, backup))
            fd, name = tempfile.mkstemp(dir=path.parent, prefix="qwen38-")
            try:
                with os.fdopen(fd, "w") as handle:
                    handle.write(after)
                shutil.copymode(path, name)
                os.replace(name, path)
            finally:
                if os.path.exists(name):
                    os.unlink(name)
    except Exception:
        for path, backup in reversed(written):
            shutil.copy2(backup, path)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vllm_root", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    changes = proposed_files(args.vllm_root.resolve())
    for path, before, after in changes:
        print("".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                          fromfile=str(path), tofile=str(path))), end="")
    if args.apply:
        apply_changes(changes)
        print("Applied integration; backups use the .qwen38.bak suffix.")


if __name__ == "__main__":
    main()
