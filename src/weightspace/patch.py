from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

SUFFIX = ".patch.npz"


@dataclass(frozen=True)
class Edit:
    """Changed positions of one tensor, flattened. Values are exact bf16 numbers held as float32."""

    indices: np.ndarray
    before: np.ndarray
    after: np.ndarray


@dataclass(frozen=True)
class Patch:
    edits: dict[str, Edit]

    @property
    def size(self) -> int:
        return sum(len(edit.indices) for edit in self.edits.values())

    def inverse(self) -> "Patch":
        return Patch({name: Edit(e.indices, e.after, e.before) for name, e in self.edits.items()})

    def save(self, path: Path) -> None:
        arrays = {f"{name}/{part}": getattr(edit, part) for name, edit in self.edits.items() for part in ("indices", "before", "after")}
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> "Patch":
        with np.load(path) as data:
            names = {key.rsplit("/", 1)[0] for key in data.files}
            return cls({name: Edit(data[f"{name}/indices"], data[f"{name}/before"], data[f"{name}/after"]) for name in sorted(names)})


def parameters(model: nn.Module) -> dict[str, mx.array]:
    return dict(tree_flatten(model.parameters()))


def as_float32(values: mx.array) -> np.ndarray:
    return np.array(values.astype(mx.float32))


def diff(baseline: Mapping[str, mx.array], edited: Mapping[str, np.ndarray | mx.array]) -> Patch:
    """The Patch that turns `baseline` tensors into `edited` ones, after rounding `edited` to the baseline's dtype."""
    edits = {}
    for name, values in edited.items():
        if name not in baseline:
            raise KeyError(f"{name} is not a tensor of this model")
        old = baseline[name]
        new = mx.array(values)
        if new.shape != old.shape:
            raise ValueError(f"{name} has shape {new.shape}, expected {old.shape}")
        if not mx.all(mx.isfinite(new)).item():
            raise ValueError(f"{name} contains NaN or infinite values")
        new = new.astype(old.dtype).reshape(-1)
        old = old.reshape(-1)
        changed = np.flatnonzero(np.array(new != old))
        if changed.size:
            where = mx.array(changed)
            edits[name] = Edit(changed, as_float32(old[where]), as_float32(new[where]))
    return Patch(edits)


def apply(model: nn.Module, patch: Patch) -> None:
    """Write the Patch into the model. Refuses if the model's weights at those positions are not the Patch's `before`."""
    params = parameters(model)
    updates = []
    for name, edit in patch.edits.items():
        tensor = params[name]
        flat = tensor.reshape(-1)
        where = mx.array(edit.indices)
        if not np.array_equal(as_float32(flat[where]), edit.before):
            raise ValueError(f"{name} does not match the Patch's starting values; was it already applied?")
        flat[where] = mx.array(edit.after).astype(tensor.dtype)
        updates.append((name, flat.reshape(tensor.shape)))
    model.update(tree_unflatten(updates))
    mx.eval(model.parameters())


@contextmanager
def patched(model: nn.Module, patch: Patch) -> Iterator[None]:
    apply(model, patch)
    try:
        yield
    finally:
        apply(model, patch.inverse())


def random_control(model: nn.Module, patch: Patch, seed: int) -> Patch:
    """The Patch's deltas moved to random positions in the same tensors, the same number per tensor.

    A delta too small to change the weight it lands on, after bf16 rounding, is dropped, so the control's size can be
    slightly below the Patch's.
    """
    rng = np.random.default_rng(seed)
    params = parameters(model)
    edits = {}
    for name in sorted(patch.edits):
        edit = patch.edits[name]
        tensor = params[name]
        indices = np.sort(rng.choice(tensor.size, size=len(edit.indices), replace=False))
        before = as_float32(tensor.reshape(-1)[mx.array(indices)])
        deltas = rng.permutation(edit.after - edit.before)
        after = as_float32(mx.array(before + deltas).astype(tensor.dtype))
        kept = after != before
        edits[name] = Edit(indices[kept], before[kept], after[kept])
    return Patch({name: edit for name, edit in edits.items() if len(edit.indices)})
