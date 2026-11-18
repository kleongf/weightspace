import json
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from weightspace.target import Target
from weightspace.tasks import TASKS

MAX_TOKENS = {False: 768, True: 2048}


@dataclass(frozen=True)
class SurveyRow:
    task: str
    thinking: bool
    n: int
    strict: float
    lenient: float
    truncated: float
    mean_tokens: float
    examples_per_second: float

    @property
    def gap(self) -> float:
        return self.lenient - self.strict


def survey(model_path: str, task_names: list[str], n: int, modes: list[bool], raw: Path, batch_size: int, sampled: bool) -> list[SurveyRow]:
    target = Target(model_path)
    rows = []
    raw.parent.mkdir(parents=True, exist_ok=True)
    with raw.open("w") as log:
        for name in task_names:
            task = TASKS[name]
            examples = task.examples("survey", n)
            for thinking in modes:
                started = time.monotonic()
                completions = target.complete([e.prompt for e in examples], thinking=thinking, max_tokens=MAX_TOKENS[thinking], batch_size=batch_size, sampled=sampled)
                seconds = time.monotonic() - started
                strict = [task.strict(e, c) for e, c in zip(examples, completions)]
                lenient = [task.lenient(e, c) for e, c in zip(examples, completions)]
                for e, c, s, l in zip(examples, completions, strict, lenient):
                    log.write(json.dumps({"task": name, "thinking": thinking, "id": e.id, "strict": s, "lenient": l, **asdict(c)}) + "\n")
                rows.append(
                    SurveyRow(
                        task=name,
                        thinking=thinking,
                        n=n,
                        strict=sum(strict) / n,
                        lenient=sum(lenient) / n,
                        truncated=sum(not c.finished for c in completions) / n,
                        mean_tokens=sum(c.tokens for c in completions) / n,
                        examples_per_second=n / seconds,
                    )
                )
                print(f"{name} thinking={thinking}: strict={rows[-1].strict:.2f} lenient={rows[-1].lenient:.2f} in {seconds:.0f}s", flush=True)
    return rows


def report(rows: list[SurveyRow], model_path: str, n: int, sampled: bool) -> str:
    lines = [
        f"# Survey of {Path(model_path).name}, {date.today().isoformat()}",
        "",
        f"{n} examples per task, {'sampled decoding (temperature 0.6, top-p 0.95, top-k 20)' if sampled else 'greedy decoding'}, token limit {MAX_TOKENS[False]} without thinking and {MAX_TOKENS[True]} with it.",
        "Survey examples come from training splits, so they never overlap a later Held-out set.",
        "",
        "| Task | Thinking | Strict | Lenient | Gap | Truncated | Mean tokens | Examples/s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: -r.gap):
        lines.append(
            f"| {r.task} | {'on' if r.thinking else 'off'} | {r.strict:.2f} | {r.lenient:.2f} | {r.gap:+.2f} "
            f"| {r.truncated:.2f} | {r.mean_tokens:.0f} | {r.examples_per_second:.2f} |"
        )
    return "\n".join(lines) + "\n"
