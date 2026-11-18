"""The trusted side of a Run: it owns the Eval budget, the Turn limit and the run record, and drives the sandbox worker."""

import json
import os
import re
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from weightspace.completion import Completion
from weightspace.tasks import TASKS, Example

REPO = Path(__file__).resolve().parents[2]
SANDBOX_MODULES = ("__init__", "completion", "target", "patch", "worker")
START_TIMEOUT = 300
SNAPSHOT_TIMEOUT = 600
GENERATE_TIMEOUT = 3600
INTERRUPT_GRACE = 30
SHOWN_CHARACTERS = 300
# Disk a Run always leaves free. A Patch that would eat into it is refused before it is written.
DISK_RESERVE = 10**9


@dataclass(frozen=True)
class RunConfig:
    task: str
    model: str
    thinking: bool
    sampled: bool
    max_tokens: int
    batch_size: int
    practice_size: int
    eval_size: int
    eval_budget: int
    turn_limit: int
    sandbox_timeout: float

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "RunConfig":
        return cls(**json.loads(path.read_text()))


def sandbox_profile(sandbox: Path) -> str:
    """A sandbox-exec profile that hides the repo (except the Target, the venv and the sandbox's own modules), the
    Hugging Face cache and Claude's transcripts; blocks the network; and allows writes only to the sandbox folder and
    the user's temporary folders."""
    repo, home = str(REPO), str(Path.home())
    package = re.escape(f"{repo}/src/weightspace")
    modules = "|".join(SANDBOX_MODULES)
    user_temp = str(Path(tempfile.gettempdir()).resolve().parent)
    return f"""(version 1)
(allow default)
(deny network*)
(deny file-read* (subpath "{repo}") (subpath "{home}/.cache/huggingface") (subpath "{home}/.claude"))
(allow file-read* (literal "{repo}") (literal "{repo}/src") (subpath "{repo}/.venv") (subpath "{repo}/models")
  (regex #"^{package}(/__pycache__)?$")
  (regex #"^{package}/(__pycache__/)?({modules})(\\.cpython-[0-9]+)?\\.pyc?$"))
(deny file-write*)
(allow file-write* (subpath "{sandbox.resolve()}") (subpath "{user_temp}") (literal "/dev/null"))
"""


def clone(source: Path, target: Path) -> None:
    """Copy a file as an APFS clone, which shares disk blocks with the source until either changes."""
    if subprocess.run(["/bin/cp", "-c", str(source), str(target)], capture_output=True).returncode:
        shutil.copy(source, target)


class WorkerDied(Exception):
    pass


class Worker:
    def __init__(self, sandbox: Path):
        worker_reads, parent_writes = os.pipe()
        parent_reads, worker_writes = os.pipe()
        with open(sandbox / "worker.log", "ab") as log:
            self.process = subprocess.Popen(
                ["/usr/bin/sandbox-exec", "-p", sandbox_profile(sandbox), sys.executable, "-m", "weightspace.worker", str(worker_reads), str(worker_writes)],
                pass_fds=(worker_reads, worker_writes),
                cwd=sandbox,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={"PATH": "/usr/bin:/bin", "HOME": str(Path.home()), "TMPDIR": tempfile.gettempdir(), "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1"},
            )
        os.close(worker_reads)
        os.close(worker_writes)
        self.requests = os.fdopen(parent_writes, "w")
        self.replies = parent_reads
        self.buffer = b""

    def request(self, message: dict, timeout: float) -> dict:
        try:
            self.requests.write(json.dumps(message) + "\n")
            self.requests.flush()
        except BrokenPipeError as error:
            raise WorkerDied from error
        return self.receive(timeout)

    def receive(self, timeout: float) -> dict:
        deadline = time.monotonic() + timeout
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            if select.select([self.replies], [], [], remaining)[0]:
                chunk = os.read(self.replies, 1 << 16)
                if not chunk:
                    raise WorkerDied
                self.buffer += chunk
        line, self.buffer = self.buffer.split(b"\n", 1)
        return json.loads(line)

    def interrupt(self) -> None:
        self.process.send_signal(signal.SIGINT)

    def close(self) -> None:
        self.requests.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        os.close(self.replies)


class Session:
    def __init__(self, run_dir: Path, config: RunConfig):
        self.run_dir = run_dir.resolve()
        self.config = config
        self.task = TASKS[config.task]
        self.practice: dict[str, Example] = {e.id: e for e in self.task.examples("practice", config.practice_size)}
        self.sandbox = self.run_dir / "sandbox"
        (self.sandbox / "evals").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "evals").mkdir(exist_ok=True)
        self.turns = 0
        self.evaluations = 0
        self.best: tuple[int, float] | None = None  # (evaluation, Practice Strict) of the only Patch kept
        self.worker: Worker | None = None

    def python(self, code: str) -> str:
        if over := self._take_turn():
            return self._log("python", {"code": code}, over)
        try:
            output = self._worker().request({"op": "exec", "code": code}, self.config.sandbox_timeout)["output"]
        except TimeoutError:
            output = self._interrupt()
        except WorkerDied:
            output = self._restart("The sandbox crashed")
        return self._log("python", {"code": code}, output)

    def evaluate(self, example_ids: list[str] | None = None) -> str:
        arguments = {"example_ids": example_ids}
        if over := self._take_turn():
            return self._log("evaluate", arguments, over)
        if self.evaluations >= self.config.eval_budget:
            return self._log("evaluate", arguments, f"The Eval budget of {self.config.eval_budget} is spent. You can still use python until the Turn limit.")
        ids = example_ids or list(self.practice)[: self.config.eval_size]
        if problem := self._check_ids(ids):
            return self._log("evaluate", arguments, f"Not evaluated, and no evaluation spent: {problem}")
        number = self.evaluations + 1
        name = f"{number:02d}.patch.npz"
        try:
            room = shutil.disk_usage(self.run_dir).free - DISK_RESERVE
            saved = self._worker().request({"op": "snapshot", "path": name, "max_bytes": room}, SNAPSHOT_TIMEOUT)
            if "error" in saved:
                return self._log("evaluate", arguments, f"Not evaluated, and no evaluation spent: {saved['error']}")
            examples = [self.practice[i] for i in ids]
            reply = self._worker().request(
                {
                    "op": "generate",
                    "prompts": [e.prompt for e in examples],
                    "thinking": self.config.thinking,
                    "max_tokens": self.config.max_tokens,
                    "batch_size": self.config.batch_size,
                    "sampled": self.config.sampled,
                },
                GENERATE_TIMEOUT,
            )
        except (TimeoutError, WorkerDied):
            return self._log("evaluate", arguments, self._restart("Evaluation failed") + " No evaluation was spent.")
        if "error" in reply:
            return self._log("evaluate", arguments, f"Not evaluated, and no evaluation spent: {reply['error']}")
        self.evaluations = number
        shutil.move(self.sandbox / name, self.run_dir / "evals" / name)
        rows = [self._score(e, Completion(**c)) for e, c in zip(examples, reply["completions"])]
        for folder in (self.run_dir, self.sandbox):
            (folder / "evals" / f"{number:02d}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        strict = sum(r["strict"] for r in rows) / len(rows)
        kept = self._keep_if_best(number, strict)
        record = {
            "evaluation": number,
            "strict": strict,
            "lenient": sum(r["lenient"] for r in rows) / len(rows),
            "patch": f"evals/{name}" if kept else None,
            "patch_size": saved["size"],
            "best": self.best[0],
        }
        return self._log("evaluate", arguments, self._report(record, rows), record)

    def _keep_if_best(self, number: int, strict: float) -> bool:
        """Keep only the Patch that can still become the final Patch: the highest Practice Strict, the later on a tie."""
        name = f"{number:02d}.patch.npz"
        if self.best is not None and strict < self.best[1]:
            (self.run_dir / "evals" / name).unlink()
            return False
        if self.best is not None:
            old = f"{self.best[0]:02d}.patch.npz"
            (self.run_dir / "evals" / old).unlink()
            (self.sandbox / "evals" / old).unlink(missing_ok=True)
        self.best = (number, strict)
        clone(self.run_dir / "evals" / name, self.sandbox / "evals" / name)
        return True

    def close(self) -> None:
        if self.worker:
            self.worker.close()
            self.worker = None

    def _worker(self) -> Worker:
        if self.worker is None:
            self.worker = Worker(self.sandbox)
            practice = [{"id": e.id, "prompt": e.prompt, "reference": e.reference} for e in self.practice.values()]
            (self.sandbox / "practice.json").write_text(json.dumps(practice, indent=1))
            reply = self.worker.request({"op": "init", "model": self.config.model, "practice": practice}, START_TIMEOUT)
            if "error" in reply:
                raise RuntimeError(f"sandbox failed to start: {reply['error']}")
        return self.worker

    def _take_turn(self) -> str | None:
        self.turns += 1
        if self.turns > self.config.turn_limit:
            return f"The Turn limit of {self.config.turn_limit} is reached. The Run is over; stop now."
        return None

    def _check_ids(self, ids: list[str]) -> str | None:
        unknown = [i for i in ids if i not in self.practice]
        if unknown:
            return f"unknown example ids {unknown[:3]}; ids look like {next(iter(self.practice))!r}"
        if len(set(ids)) != len(ids):
            return "example ids repeat"
        if len(ids) > self.config.eval_size:
            return f"at most {self.config.eval_size} examples per evaluation"
        return None

    def _interrupt(self) -> str:
        self.worker.interrupt()
        try:
            return self.worker.receive(INTERRUPT_GRACE)["output"]
        except (TimeoutError, WorkerDied):
            return self._restart(f"The cell ran past {self.config.sandbox_timeout:.0f} seconds and would not stop")

    def _restart(self, why: str) -> str:
        if self.worker:
            self.worker.process.kill()
            self.worker.close()
            self.worker = None
        return (
            f"{why}, so the sandbox restarts from the Baseline weights on your next call and your edits are lost. "
            "The best evaluated Patch so far is saved in evals/ in your sandbox folder; reapply it with apply(model, Patch.load(path))."
        )

    def _score(self, example: Example, completion: Completion) -> dict:
        return {
            "id": example.id,
            "strict": self.task.strict(example, completion),
            "lenient": self.task.lenient(example, completion),
            **asdict(completion),
        }

    def _report(self, record: dict, rows: list[dict]) -> str:
        left = self.config.eval_budget - record["evaluation"]
        cut = sum(not r["finished"] for r in rows)
        lines = [
            f"Evaluation {record['evaluation']} of {self.config.eval_budget} ({left} left). "
            f"Strict {record['strict']:.2f}, Lenient {record['lenient']:.2f}, cut off {cut}/{len(rows)}, "
            f"Patch size {record['patch_size']:,} weights."
        ]
        for r in rows:
            shown = r["answer"] if r["finished"] and r["answer"] is not None else f"(cut off) ...{r['text'][-SHOWN_CHARACTERS:]}"
            lines.append(f"{r['id']} strict={r['strict']:.0f} lenient={r['lenient']:.2f} tokens={r['tokens']}: {shown[:SHOWN_CHARACTERS]!r}")
        if record["patch"]:
            lines.append(f"Patch kept as {record['patch']}, the best so far; earlier Patches are deleted.")
        else:
            lines.append(f"Patch not kept: evaluation {record['best']} scored higher, and its Patch stays in evals/.")
        lines.append(f"Full outputs: evals/{record['evaluation']:02d}.jsonl in your sandbox folder.")
        return "\n".join(lines)

    def _log(self, tool: str, arguments: dict, output: str, record: dict | None = None) -> str:
        entry = {"turn": self.turns, "time": time.time(), "tool": tool, "input": arguments, "output": output, **(record or {})}
        with open(self.run_dir / "log.jsonl", "a") as log:
            log.write(json.dumps(entry) + "\n")
        return output
