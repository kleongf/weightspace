import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from weightspace.patch import Patch, apply, diff, parameters, patched, random_control
from weightspace.stats import Interval, paired_bootstrap, verdict


@pytest.fixture
def model():
    mx.random.seed(0)
    layer = nn.Linear(16, 16)
    layer.set_dtype(mx.bfloat16)
    return layer


def weights(model) -> np.ndarray:
    return np.array(model.weight.astype(mx.float32))


def edited(model, **changes) -> dict[str, np.ndarray]:
    w = weights(model)
    for position, value in changes.items():
        w.flat[int(position.removeprefix("at"))] = value
    return {"weight": w}


def test_unchanged_weights_give_an_empty_patch(model):
    assert diff(parameters(model), {"weight": weights(model)}).size == 0


def test_changes_below_bf16_precision_are_not_edits(model):
    w = weights(model)
    w.flat[3] += 1e-6
    assert diff(parameters(model), {"weight": w}).size == 0


def test_patch_records_exactly_the_changed_positions(model):
    patch = diff(parameters(model), edited(model, at3=2.0, at40=-1.5))
    assert patch.size == 2
    assert patch.edits["weight"].indices.tolist() == [3, 40]
    assert patch.edits["weight"].after.tolist() == [2.0, -1.5]


def test_applying_a_patch_changes_the_output_and_restoring_undoes_it(model):
    x = mx.ones((1, 16), dtype=mx.bfloat16)
    original = np.array(model(x).astype(mx.float32))
    with patched(model, diff(parameters(model), edited(model, at0=5.0))):
        assert not np.array_equal(np.array(model(x).astype(mx.float32)), original)
    assert np.array_equal(np.array(model(x).astype(mx.float32)), original)


def test_a_patch_cannot_be_applied_twice(model):
    patch = diff(parameters(model), edited(model, at0=5.0))
    apply(model, patch)
    with pytest.raises(ValueError, match="already applied"):
        apply(model, patch)


def test_non_finite_edits_are_refused(model):
    with pytest.raises(ValueError, match="NaN"):
        diff(parameters(model), edited(model, at0=np.nan))


def test_patch_survives_saving_and_loading(model, tmp_path):
    patch = diff(parameters(model), edited(model, at3=2.0, at40=-1.5))
    patch.save(tmp_path / "p.patch.npz")
    loaded = Patch.load(tmp_path / "p.patch.npz")
    assert loaded.size == 2
    for part in ("indices", "before", "after"):
        assert np.array_equal(getattr(loaded.edits["weight"], part), getattr(patch.edits["weight"], part))


def test_random_control_moves_the_same_deltas_to_new_positions(model):
    patch = diff(parameters(model), edited(model, at3=4.0, at40=-4.0, at100=8.0))
    control = random_control(model, patch, seed=1)
    assert control.edits.keys() == patch.edits.keys()
    assert control.size == patch.size
    assert set(control.edits["weight"].indices) != set(patch.edits["weight"].indices)
    before = parameters(model)["weight"]
    with patched(model, control):
        changed = np.flatnonzero(weights(model) != np.array(before.astype(mx.float32)))
    assert changed.tolist() == control.edits["weight"].indices.tolist()


def test_identical_scores_show_no_difference():
    scores = [1.0, 0.0, 1.0, 1.0, 0.0] * 20
    assert paired_bootstrap(scores, scores) == Interval(0.0, 0.0, 0.0)


def test_verdict_compares_against_the_best_control():
    patch = [1.0] * 80 + [0.0] * 20
    weak, strong = [0.0] * 100, [1.0] * 70 + [0.0] * 30
    result = verdict(patch, [weak, strong])
    assert result.best_control == 1
    assert result.difference.mean == pytest.approx(0.1)
    assert result.passed


def test_verdict_fails_when_the_patch_ties_a_control():
    scores = [1.0, 0.0] * 50
    assert not verdict(scores, [scores]).passed
