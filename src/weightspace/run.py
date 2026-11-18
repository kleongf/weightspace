"""Launch one Run: a headless Claude Code Editor, outside the repo, whose only tools come from the weightspace MCP server.

The run folder lives under RUNS_ROOT while the Editor works, then its record is copied into the Experiment's runs/.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from string import Template

from weightspace.session import REPO, RunConfig

RUNS_ROOT = Path.home() / "weightspace-runs"
EDITOR_TOOLS = ("mcp__weightspace__python", "mcp__weightspace__evaluate")
TOOL_TIMEOUT_MS = 2 * 60 * 60 * 1000
DRY = dict(eval_budget=1, eval_size=2, turn_limit=2, max_tokens=256, practice_size=16)
RECORD = ("config.json", "launch.json", "prompt.md", "log.jsonl", "transcript.jsonl", "claude.log", "evals", "sandbox/worker.log")


@dataclass(frozen=True)
class Plan:
    editor_model: str
    spend_cap_usd: float
    session: RunConfig

    @classmethod
    def load(cls, path: Path) -> "Plan":
        raw = json.loads(path.read_text())
        session = RunConfig(**raw["session"])
        session = replace(session, model=str((REPO / session.model).resolve()))
        return cls(raw["editor_model"], float(raw["spend_cap_usd"]), session)


def prompt(experiment: Path, config: RunConfig) -> str:
    fields = {**asdict(config), "sandbox_minutes": f"{config.sandbox_timeout / 60:.0f}"}
    return Template((experiment / "editor.md").read_text()).substitute(fields)


def next_name(folder: Path, prefix: str) -> str:
    taken = {p.name for p in folder.iterdir()} if folder.is_dir() else set()
    return next(f"{prefix}{n:02d}" for n in range(1, 100) if f"{prefix}{n:02d}" not in taken)


def command(text: str, plan: Plan, mcp_config: Path) -> list[str]:
    return [
        "claude", "-p", text,
        "--model", plan.editor_model,
        "--tools", "",
        "--strict-mcp-config", "--mcp-config", str(mcp_config),
        "--allowedTools", *EDITOR_TOOLS,
        "--max-budget-usd", str(plan.spend_cap_usd),
        "--setting-sources", "",
        "--disable-slash-commands",
        "--no-session-persistence",
        "--output-format", "stream-json", "--verbose",
    ]  # fmt: skip


def git_state() -> dict:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True).stdout.strip())
    return {"commit": head, "dirty": dirty}


def launch(experiment: Path, plan: Plan, dry: bool) -> tuple[Path, dict]:
    """Run the Editor to completion and return the run folder and Claude's final result message."""
    if dry:
        plan = replace(plan, session=replace(plan.session, **DRY))
    name = next_name(RUNS_ROOT / experiment.name, "dry-" if dry else "")
    run_dir = RUNS_ROOT / experiment.name / name
    editor_dir = run_dir / "editor"
    editor_dir.mkdir(parents=True)
    plan.session.save(run_dir / "config.json")
    text = prompt(experiment, plan.session)
    (run_dir / "prompt.md").write_text(text)
    mcp_config = run_dir / "mcp.json"
    server = {"command": sys.executable, "args": ["-m", "weightspace.server", str(run_dir)]}
    mcp_config.write_text(json.dumps({"mcpServers": {"weightspace": server}}, indent=2))
    argv = command(text, plan, mcp_config)
    meta = {"experiment": experiment.name, "run": name, "dry": dry, "started": time.time(), **git_state(), "argv": argv[:2] + ["<prompt.md>"] + argv[3:]}
    (run_dir / "launch.json").write_text(json.dumps(meta, indent=2))
    env = {k: v for k, v in os.environ.items() if not k.startswith("ANTHROPIC_")}
    env["MCP_TOOL_TIMEOUT"] = str(TOOL_TIMEOUT_MS)
    with open(run_dir / "transcript.jsonl", "w") as transcript, open(run_dir / "claude.log", "w") as log:
        subprocess.run(argv, cwd=editor_dir, env=env, stdin=subprocess.DEVNULL, stdout=transcript, stderr=log)
    if not dry:
        keep(run_dir, experiment / "runs" / name)
    return run_dir, result(run_dir / "transcript.jsonl")


def keep(run_dir: Path, record: Path) -> None:
    record.mkdir(parents=True)
    for part in RECORD:
        source = run_dir / part
        if source.is_dir():
            shutil.copytree(source, record / part)
        elif source.is_file():
            (record / part).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source, record / part)


def result(transcript: Path) -> dict:
    lines = [json.loads(line) for line in transcript.read_text().splitlines() if line.strip()]
    return next((m for m in reversed(lines) if m.get("type") == "result"), {})
