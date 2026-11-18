"""Score a finished Run's final Patch on the Held-out set, against its Random-edit controls and the Baseline.

This runs on the trusted side only: it generates the Held-out examples, which never reach a run folder (ADR 0002).
Each scoring is saved to its own file as soon as it finishes, so an interrupted scoring resumes where it stopped.
"""

import hashlib
import json
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from weightspace.completion import Completion
from weightspace.patch import Patch, patched, random_control
from weightspace.session import RunConfig
from weightspace.stats import paired_bootstrap, verdict
from weightspace.tasks import TASKS, Example


@dataclass(frozen=True)
class ScorePlan:
    held_out: int
    controls: int
    carry_over_task: str
    carry_over: int

    @classmethod
    def load(cls, run_json: Path) -> "ScorePlan":
        return cls(**json.loads(run_json.read_text())["scoring"])


class Scorer(Protocol):
    model: object

    def complete(self, prompts: list[str], *, thinking: bool, max_tokens: int, batch_size: int, sampled: bool) -> list[Completion]: ...


def final_patch(record: Path) -> tuple[int, Patch]:
    """The evaluation the Run's log names as best, and its Patch, from the record or from where its pointer says."""
    entries = [json.loads(line) for line in (record / "log.jsonl").read_text().splitlines() if line.strip()]
    best = [e["best"] for e in entries if "best" in e]
    if not best:
        raise ValueError(f"{record.name} has no evaluated Patch to score")
    number = best[-1]
    kept = record / "evals" / f"{number:02d}.patch.npz"
    if kept.is_file():
        return number, Patch.load(kept)
    pointer = json.loads((record / "evals" / f"{number:02d}.patch.json").read_text())
    path = Path(pointer["path"])
    with open(path, "rb") as data:
        if hashlib.file_digest(data, "sha256").hexdigest() != pointer["sha256"]:
            raise ValueError(f"{path} does not match the SHA-256 in {record.name}'s record")
    return number, Patch.load(path)


def scored(path: Path, target: Scorer, config: RunConfig, task_name: str, examples: list[Example], patch: Patch | None) -> list[dict]:
    """Per-example scores of the Target with `patch` applied, read from `path` if this scoring already finished."""
    if path.is_file():
        return [json.loads(line) for line in path.read_text().splitlines()]
    task = TASKS[task_name]
    started = time.monotonic()
    with patched(target.model, patch) if patch is not None else nullcontext():
        completions = target.complete(
            [e.prompt for e in examples], thinking=config.thinking, max_tokens=config.max_tokens, batch_size=config.batch_size, sampled=config.sampled
        )
    rows = [{"id": e.id, "strict": task.strict(e, c), "lenient": task.lenient(e, c), **asdict(c)} for e, c in zip(examples, completions)]
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    partial.write_text("".join(json.dumps(row) + "\n" for row in rows))
    os.replace(partial, path)
    print(f"{path.parent.name}/{path.stem}: Strict {summary(rows)['strict']:.2f} in {time.monotonic() - started:.0f}s", flush=True)
    return rows


def summary(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "strict": float(np.mean([r["strict"] for r in rows])),
        "lenient": float(np.mean([r["lenient"] for r in rows])),
        "cut_off": float(np.mean([not r["finished"] for r in rows])),
        "looped": float(np.mean([r["looped"] for r in rows])),
        "mean_tokens": float(np.mean([r["tokens"] for r in rows])),
    }


def baseline_folder(record: Path, config: RunConfig, plan: ScorePlan) -> Path:
    """The Experiment's Baseline scorings, shared by its Runs, refused if they were made with other settings."""
    folder = record.parent.parent / "baseline"
    settings = {key: getattr(config, key) for key in ("task", "model", "thinking", "sampled", "max_tokens", "batch_size")} | asdict(plan)
    saved = folder / "settings.json"
    if saved.is_file() and json.loads(saved.read_text()) != settings:
        raise ValueError(f"the Baseline in {folder} was scored with other settings than {record.name}; move it aside to rescore")
    folder.mkdir(exist_ok=True)
    saved.write_text(json.dumps(settings, indent=2) + "\n")
    return folder


def strict(rows: list[dict]) -> list[float]:
    return [r["strict"] for r in rows]


def score_run(record: Path, plan: ScorePlan, target: Scorer) -> dict:
    """Score the Run's final Patch, its Random-edit controls and the Baseline, and write score.json into the record."""
    config = RunConfig.load(record / "config.json")
    number, patch = final_patch(record)
    controls = [random_control(target.model, patch, seed) for seed in range(plan.controls)]
    baseline = baseline_folder(record, config, plan)
    scores = record / "scores"
    held_out = TASKS[config.task].examples("held-out", plan.held_out)
    carry_over = TASKS[plan.carry_over_task].examples("held-out", plan.carry_over)

    rows = {
        "baseline": scored(baseline / "held-out.jsonl", target, config, config.task, held_out, None),
        "patch": scored(scores / "held-out" / "patch.jsonl", target, config, config.task, held_out, patch),
    }
    for seed, control in enumerate(controls):
        rows[f"control-{seed}"] = scored(scores / "held-out" / f"control-{seed}.jsonl", target, config, config.task, held_out, control)
    carry = {
        "baseline": scored(baseline / "carry-over.jsonl", target, config, plan.carry_over_task, carry_over, None),
        "patch": scored(scores / "carry-over" / "patch.jsonl", target, config, plan.carry_over_task, carry_over, patch),
    }

    against_controls = verdict(strict(rows["patch"]), [strict(rows[f"control-{s}"]) for s in range(plan.controls)])
    result = {
        "run": record.name,
        "plan": asdict(plan),
        "final_patch": {"evaluation": number, "size": patch.size, "tensors": len(patch.edits)},
        "controls": [{"seed": seed, "size": control.size} for seed, control in enumerate(controls)],
        "held_out": {name: summary(r) for name, r in rows.items()},
        "verdict": {"best_control": against_controls.best_control, **asdict(against_controls.difference), "passed": against_controls.passed},
        "vs_baseline": asdict(paired_bootstrap(strict(rows["patch"]), strict(rows["baseline"]))),
        "carry_over": {
            "task": plan.carry_over_task,
            **{name: summary(r) for name, r in carry.items()},
            "difference": asdict(paired_bootstrap(strict(carry["patch"]), strict(carry["baseline"]))),
        },
    }
    (record / "score.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def describe(result: dict) -> str:
    held, test, base, carry = result["held_out"], result["verdict"], result["vs_baseline"], result["carry_over"]
    best = f"control-{test['best_control']}"
    interval = "{mean:+.2f} [{low:+.2f}, {high:+.2f}]".format
    return "\n".join(
        [
            f"{result['run']}: final Patch from evaluation {result['final_patch']['evaluation']}, {result['final_patch']['size']:,} weights",
            f"Held-out Strict: Patch {held['patch']['strict']:.2f}, best control ({best}) {held[best]['strict']:.2f}, Baseline {held['baseline']['strict']:.2f}",
            f"Patch minus best control: {interval(**test)}, {'passes' if test['passed'] else 'does not pass'} (needs the low end above 0)",
            f"Patch minus Baseline: {interval(**base)}",
            f"Carry-over ({carry['task']}) Strict: Patch {carry['patch']['strict']:.2f}, Baseline {carry['baseline']['strict']:.2f}, "
            f"difference {interval(**carry['difference'])}",
        ]
    )
