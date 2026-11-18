"""The Editor's sandbox process.

It runs under sandbox-exec (see `session.sandbox_profile`) and imports only modules that cannot produce the Held-out set.
The trusted Session sends one JSON message per line on one pipe and reads one JSON reply per line on another.
"""

import ast
import contextlib
import io
import json
import os
import signal
import sys
import traceback
from dataclasses import asdict

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm import load

from weightspace.patch import Patch, apply, diff, parameters
from weightspace.target import Target

OUTPUT_LIMIT = 20_000
GRADIENT_FUNCTIONS = [(mx, "grad"), (mx, "value_and_grad"), (mx, "vjp"), (mx, "jvp"), (nn, "value_and_grad")]
busy = False


def on_interrupt(signum, frame):
    if busy:
        raise KeyboardInterrupt


def ban_gradients() -> None:
    for module, name in GRADIENT_FUNCTIONS:

        def refuse(*args, _name=name, **kwargs):
            raise PermissionError(f"{_name} is banned: this Experiment allows no gradients")

        setattr(module, name, refuse)


def clip(text: str) -> str:
    if len(text) <= OUTPUT_LIMIT:
        return text
    half = OUTPUT_LIMIT // 2
    return f"{text[:half]}\n... {len(text) - OUTPUT_LIMIT} characters cut ...\n{text[-half:]}"


def run_cell(code: str, namespace: dict) -> str:
    """Run code like a notebook cell: print output, and show the value of a final expression."""
    global busy
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        busy = True
        try:
            tree = ast.parse(code, "<cell>")
            last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
            exec(compile(tree, "<cell>", "exec"), namespace)
            if last is not None:
                value = eval(compile(ast.Expression(last.value), "<cell>", "eval"), namespace)
                if value is not None:
                    print(repr(value))
        except KeyboardInterrupt:
            print("Interrupted at the sandbox timeout. Edits made before the interrupt are kept.")
        except BaseException:
            traceback.print_exc()
        finally:
            busy = False
    return clip(out.getvalue())


def snapshot(model: nn.Module, model_path: str, path: str, max_bytes: int) -> dict:
    """Save the Patch from the Baseline to the live weights, unless its uncompressed size passes `max_bytes`.
    The Baseline is reloaded lazily, one tensor at a time."""
    live = parameters(model)
    baseline = dict(tree_flatten(load(model_path, lazy=True)[0].parameters()))
    if live.keys() != baseline.keys():
        changed = sorted(live.keys() ^ baseline.keys())
        raise ValueError(f"tensors were added or removed, starting with {changed[:3]}; only weight values may change")
    edits = {}
    for name in sorted(baseline):
        old, new = baseline.pop(name), live[name]
        if new.shape != old.shape or new.dtype != old.dtype:
            raise ValueError(f"{name} must keep shape {old.shape} and dtype {old.dtype}, but it is {new.shape} {new.dtype}")
        edits.update(diff({name: old}, {name: new}).edits)
    patch = Patch(edits)
    needed = sum(part.nbytes for edit in edits.values() for part in (edit.indices, edit.before, edit.after))
    if needed > max_bytes:
        raise ValueError(
            f"this Patch changes {patch.size:,} weights and needs up to {needed / 1e9:.1f} GB of disk, "
            f"but only {max(max_bytes, 0) / 1e9:.1f} GB can be spared; make a smaller edit"
        )
    patch.save(path)
    return {"size": patch.size, "tensors": len(edits)}


def main() -> None:
    signal.signal(signal.SIGINT, on_interrupt)
    inbox = os.fdopen(int(sys.argv[1]), "r")
    outbox = os.fdopen(int(sys.argv[2]), "w")
    target: Target | None = None
    model_path = ""
    namespace: dict = {}
    for line in inbox:
        message = json.loads(line)
        try:
            op = message["op"]
            if op == "init":
                model_path = message["model"]
                target = Target(model_path)
                ban_gradients()
                namespace = {
                    "model": target.model,
                    "tokenizer": target.tokenizer,
                    "target": target,
                    "practice": message["practice"],
                    "mx": mx,
                    "nn": nn,
                    "np": np,
                    "tree_flatten": tree_flatten,
                    "tree_unflatten": tree_unflatten,
                    "Patch": Patch,
                    "apply": apply,
                }
                reply = {"parameters": sum(v.size for v in parameters(target.model).values())}
            elif op == "exec":
                reply = {"output": run_cell(message["code"], namespace)}
            elif op == "snapshot":
                reply = snapshot(target.model, model_path, message["path"], message["max_bytes"])
            elif op == "generate":
                completions = target.complete(
                    message["prompts"],
                    thinking=message["thinking"],
                    max_tokens=message["max_tokens"],
                    batch_size=message["batch_size"],
                    sampled=message["sampled"],
                )
                reply = {"completions": [asdict(c) for c in completions]}
            else:
                reply = {"error": f"unknown op {op!r}"}
        except Exception as error:
            reply = {"error": f"{type(error).__name__}: {error}"}
        outbox.write(json.dumps(reply) + "\n")
        outbox.flush()


if __name__ == "__main__":
    main()
