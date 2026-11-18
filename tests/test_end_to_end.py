"""Runs the real Target, so it takes minutes. Run with: uv run pytest -m slow"""

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from weightspace.patch import diff, parameters, patched, random_control
from weightspace.stats import paired_bootstrap
from weightspace.target import Target
from weightspace.tasks import TASKS

MODEL = Path("models/Qwen3.5-0.8B")
pytestmark = [pytest.mark.slow, pytest.mark.skipif(not MODEL.is_dir(), reason="Target weights not downloaded")]


@pytest.fixture(scope="module")
def target():
    return Target(str(MODEL))


def strict_scores(target, examples):
    task = TASKS["json-extraction"]
    completions = target.complete([e.prompt for e in examples], thinking=True, max_tokens=2048, sampled=True)
    return [task.strict(e, c) for e, c in zip(examples, completions)]


def logits(target):
    return np.array(target.model(mx.array([target.encode("Hello", thinking=False)]))[0, -1].astype(mx.float32))


def test_do_nothing_patch_scores_exactly_like_the_baseline(target):
    examples = TASKS["json-extraction"].examples("practice", 8)
    name = "language_model.model.layers.0.mlp.gate_proj.weight"
    nothing = diff(target.model, {name: parameters(target.model)[name]})
    assert nothing.size == 0
    baseline = strict_scores(target, examples)
    with patched(target.model, nothing):
        patched_scores = strict_scores(target, examples)
    assert patched_scores == baseline
    assert paired_bootstrap(patched_scores, baseline) == paired_bootstrap(baseline, baseline)


def test_a_random_patch_is_fully_undone(target):
    name = "language_model.model.layers.0.mlp.gate_proj.weight"
    before = logits(target)
    tensor = np.array(parameters(target.model)[name].astype(mx.float32))
    tensor.flat[np.arange(0, tensor.size, 997)] += 0.5
    patch = diff(target.model, {name: tensor})
    control = random_control(target.model, patch, seed=0)
    for edit in (patch, control):
        with patched(target.model, edit):
            assert not np.array_equal(logits(target), before)
        assert np.array_equal(logits(target), before)
