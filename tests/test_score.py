"""Scoring a finished Run on the Held-out set, with a fake Target, so it takes seconds."""

import hashlib
import json

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

import weightspace.tasks
from weightspace.completion import Completion
from weightspace.patch import diff, parameters
from weightspace.score import ScorePlan, score_run
from weightspace.session import RunConfig
from weightspace.tasks import Example, Task

FIXED_AT = 0


class FakeTarget:
    """Finishes every answer exactly when weight 0 is above 5, the Fault a good Patch repairs."""

    def __init__(self):
        mx.random.seed(0)
        self.model = nn.Linear(64, 64)
        self.model.set_dtype(mx.bfloat16)
        self.calls = 0

    def complete(self, prompts, *, thinking, max_tokens, batch_size, sampled):
        self.calls += 1
        fixed = self.model.weight.reshape(-1)[FIXED_AT].item() > 5
        return [Completion(text=p, answer=p if fixed else None, finished=fixed, tokens=3 if fixed else max_tokens) for p in prompts]


def fake_examples(split, n):
    return [Example(f"fake-{split}-{i}", f"prompt {split} {i}", i) for i in range(n)]


@pytest.fixture(autouse=True)
def fake_task(monkeypatch):
    task = Task("fake", fake_examples, lambda e, c: float(c.finished), lambda e, c: 1.0)
    monkeypatch.setitem(weightspace.tasks.TASKS, "fake", task)


PLAN = ScorePlan(held_out=20, controls=3, carry_over_task="fake", carry_over=10)


def make_record(tmp_path, target, run="claude-01", value=9.0, evaluations=(0.5, 0.75, 0.25)):
    """A run record whose best evaluation (the second) kept a Patch that sets weight 0 to `value`."""
    record = tmp_path / "0009-fake" / "runs" / run
    (record / "evals").mkdir(parents=True)
    RunConfig("fake", "unused", True, True, 64, 4, 16, 16, 30, 60, 600).save(record / "config.json")
    weights = np.array(target.model.weight.astype(mx.float32))
    weights.flat[FIXED_AT] = value
    diff(parameters(target.model), {"weight": weights}).save(record / "evals" / "02.patch.npz")
    best = []
    with open(record / "log.jsonl", "w") as log:
        for number, strict in enumerate(evaluations, 1):
            best.append(number if not best or strict >= evaluations[best[-1] - 1] else best[-1])
            log.write(json.dumps({"tool": "evaluate", "evaluation": number, "strict": strict, "best": best[-1]}) + "\n")
    return record


def test_a_patch_that_repairs_the_fault_beats_its_controls(tmp_path):
    target = FakeTarget()
    result = score_run(make_record(tmp_path, target), PLAN, target)
    assert result["final_patch"]["evaluation"] == 2
    assert result["held_out"]["patch"]["strict"] == 1.0
    assert all(result["held_out"][f"control-{seed}"]["strict"] == 0.0 for seed in range(3))
    assert result["verdict"]["passed"]
    assert result["vs_baseline"]["mean"] == 1.0
    assert result["carry_over"]["patch"]["strict"] == 1.0


def test_a_patch_that_changes_nothing_useful_does_not_pass(tmp_path):
    target = FakeTarget()
    result = score_run(make_record(tmp_path, target, value=-9.0), PLAN, target)
    assert not result["verdict"]["passed"]


def test_scoring_leaves_the_target_at_its_baseline(tmp_path):
    target = FakeTarget()
    before = np.array(target.model.weight.astype(mx.float32))
    score_run(make_record(tmp_path, target), PLAN, target)
    assert np.array_equal(np.array(target.model.weight.astype(mx.float32)), before)


def test_an_interrupted_scoring_resumes_without_redoing_finished_parts(tmp_path):
    target = FakeTarget()
    record = make_record(tmp_path, target)
    first = score_run(record, PLAN, target)
    calls = target.calls
    assert calls == 2 + 1 + PLAN.controls + 1  # Baseline twice, the Patch, its controls, carry-over
    (record / "scores" / "held-out" / "control-1.jsonl").unlink()
    again = score_run(record, PLAN, target)
    assert target.calls == calls + 1
    assert again == first


def test_the_baseline_is_scored_once_per_experiment(tmp_path):
    target = FakeTarget()
    score_run(make_record(tmp_path, target, "claude-01"), PLAN, target)
    calls = target.calls
    score_run(make_record(tmp_path, target, "codex-01"), PLAN, target)
    assert target.calls - calls == 1 + PLAN.controls + 1


def test_a_baseline_scored_with_other_settings_is_refused(tmp_path):
    target = FakeTarget()
    score_run(make_record(tmp_path, target, "claude-01"), PLAN, target)
    with pytest.raises(ValueError, match="Baseline"):
        score_run(make_record(tmp_path, target, "codex-01"), ScorePlan(30, 3, "fake", 10), target)


def test_a_patch_kept_outside_the_repo_is_found_by_its_record_and_checked(tmp_path):
    target = FakeTarget()
    record = make_record(tmp_path, target)
    outside = tmp_path / "outside.patch.npz"
    (record / "evals" / "02.patch.npz").rename(outside)
    digest = hashlib.sha256(outside.read_bytes()).hexdigest()
    pointer = record / "evals" / "02.patch.json"
    pointer.write_text(json.dumps({"path": str(outside), "bytes": outside.stat().st_size, "sha256": digest}))
    assert score_run(record, PLAN, target)["verdict"]["passed"]
    pointer.write_text(json.dumps({"path": str(outside), "bytes": outside.stat().st_size, "sha256": "0" * 64}))
    (record / "scores" / "held-out" / "patch.jsonl").unlink()
    with pytest.raises(ValueError, match="SHA-256"):
        score_run(record, PLAN, target)


def test_a_run_that_never_evaluated_has_no_final_patch_to_score(tmp_path):
    target = FakeTarget()
    record = make_record(tmp_path, target, evaluations=())
    with pytest.raises(ValueError, match="no evaluated Patch"):
        score_run(record, PLAN, target)
