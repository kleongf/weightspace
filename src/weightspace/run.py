"""Launch one Run: a headless Editor, outside the repo, whose only tools come from the weightspace MCP server.

`ws` starts the trusted server itself, on a secret local URL, then runs one Editor CLI against it. The run folder lives
under RUNS_ROOT while the Editor works, then its record is copied into the Experiment's runs/.
"""

import hashlib
import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from string import Template
from typing import Iterator, Protocol

from weightspace.session import REPO, RunConfig

RUNS_ROOT = Path.home() / "weightspace-runs"
TOOL_TIMEOUT_SECONDS = 2 * 60 * 60
# Keeps the Mac awake while the Editor runs. On battery with the lid closed it still sleeps; outcome.json records how long.
STAY_AWAKE = ("/usr/bin/caffeinate", "-i", "-s")
SERVER_START_SECONDS = 60
DRY = dict(eval_budget=1, eval_size=2, turn_limit=2, max_tokens=256, practice_size=16)
# A real Run needs the Mac on power and room for two copies of a large Patch.
MIN_FREE_BYTES = 3 * 10**9
# Larger Patches stay in RUNS_ROOT; the repo's record names them by size and hash. GitHub refuses files over 100 MB.
GIT_FILE_LIMIT = 50 * 10**6
RECORD = ("config.json", "launch.json", "outcome.json", "prompt.md", "log.jsonl", "transcript.jsonl", "editor.log", "server.log", "evals", "sandbox/worker.log")


@dataclass(frozen=True)
class EditorSettings:
    model: str
    effort: str
    spend_cap_usd: float | None = None


@dataclass(frozen=True)
class Plan:
    editors: dict[str, EditorSettings]
    session: RunConfig

    @classmethod
    def load(cls, path: Path) -> "Plan":
        raw = json.loads(path.read_text())
        session = RunConfig(**raw["session"])
        session = replace(session, model=str((REPO / session.model).resolve()))
        return cls({name: EditorSettings(**settings) for name, settings in raw["editors"].items()}, session)


@dataclass(frozen=True)
class Launch:
    """Everything an Editor needs to start: its settings, prompt, run folder and the server's URL."""

    settings: EditorSettings
    prompt: str
    run_dir: Path
    url: str

    @property
    def workdir(self) -> Path:
        return self.run_dir / "editor"


class Editor(Protocol):
    def command(self, launch: Launch) -> tuple[list[str], dict[str, str]]: ...

    def outcome(self, transcript: Path) -> dict: ...

    def finish(self, launch: Launch) -> None: ...


def clean_env(**extra: str) -> dict[str, str]:
    """The parent environment without API keys, so every Editor runs on the user's subscription."""
    keys = ("ANTHROPIC_", "OPENAI_API_KEY", "CODEX_API_KEY")
    return {**{k: v for k, v in os.environ.items() if not k.startswith(keys)}, **extra}


def transcript_lines(transcript: Path) -> list[dict]:
    return [json.loads(line) for line in transcript.read_text().splitlines() if line.strip()] if transcript.is_file() else []


class Claude:
    TOOLS = ("mcp__weightspace__python", "mcp__weightspace__evaluate")

    def command(self, launch: Launch) -> tuple[list[str], dict[str, str]]:
        mcp_config = launch.run_dir / "mcp.json"
        mcp_config.write_text(json.dumps({"mcpServers": {"weightspace": {"type": "http", "url": launch.url}}}, indent=2))
        cap = [] if launch.settings.spend_cap_usd is None else ["--max-budget-usd", str(launch.settings.spend_cap_usd)]
        argv = [
            "claude", "-p", launch.prompt,
            "--model", launch.settings.model,
            "--effort", launch.settings.effort,
            "--tools", "",
            "--strict-mcp-config", "--mcp-config", str(mcp_config),
            "--allowedTools", *self.TOOLS,
            *cap,
            "--setting-sources", "",
            "--disable-slash-commands",
            "--no-session-persistence",
            "--output-format", "stream-json", "--verbose",
        ]  # fmt: skip
        return argv, clean_env(MCP_TOOL_TIMEOUT=str(TOOL_TIMEOUT_SECONDS * 1000))

    def outcome(self, transcript: Path) -> dict:
        result = next((m for m in reversed(transcript_lines(transcript)) if m.get("type") == "result"), {})
        usage = result.get("usage", {})
        cached = usage.get("cache_read_input_tokens", 0)
        return {
            "status": result.get("subtype", "no result"),
            "cost_usd": result.get("total_cost_usd"),
            "input_tokens": usage.get("input_tokens", 0) + usage.get("cache_creation_input_tokens", 0) + cached,
            "cached_input_tokens": cached,
            "output_tokens": usage.get("output_tokens", 0),
        }

    def finish(self, launch: Launch) -> None:
        pass


class Codex:
    """Codex CLI in exec mode. It keeps a few built-in tools that cannot be switched off (apply_patch, view_image and
    the code-mode host), so it runs under a sandbox-exec profile that hides everything a Run must not see."""

    APP_BINARY = Path("/Applications/ChatGPT.app/Contents/Resources/codex-cli/bin/codex")
    AUTH = Path.home() / ".codex" / "auth.json"

    def binary(self) -> str:
        found = os.environ.get("WS_CODEX") or shutil.which("codex") or (str(self.APP_BINARY) if self.APP_BINARY.is_file() else None)
        if not found:
            raise FileNotFoundError("Codex CLI not found; set WS_CODEX to its path")
        return found

    def command(self, launch: Launch) -> tuple[list[str], dict[str, str]]:
        home = launch.run_dir / "codex-home"
        home.mkdir()
        (home / "config.toml").write_text(self.config(launch))
        (home / "auth.json").symlink_to(self.AUTH)
        argv = [
            "/usr/bin/sandbox-exec", "-p", codex_profile(launch.run_dir),
            self.binary(), "exec", "--json", "--ephemeral", "--skip-git-repo-check", "-s", "read-only",
            "-m", launch.settings.model, launch.prompt,
        ]  # fmt: skip
        return argv, clean_env(CODEX_HOME=str(home))

    def config(self, launch: Launch) -> str:
        off = (
            "shell_tool", "unified_exec", "apps", "plugins", "hooks", "image_generation", "browser_use", "in_app_browser",
            "computer_use", "sleep_tool", "skill_search", "tool_suggest", "goals", "memories",
        )  # fmt: skip
        features = "\n".join(f"{name} = false" for name in off)
        return f"""model = "{launch.settings.model}"
model_reasoning_effort = "{launch.settings.effort}"
web_search = "disabled"
include_permissions_instructions = false

[agents]
enabled = false

[skills.bundled]
enabled = false

[features]
{features}

[mcp_servers.weightspace]
url = "{launch.url}"
tool_timeout_sec = {TOOL_TIMEOUT_SECONDS}
default_tools_approval_mode = "approve"
"""

    def outcome(self, transcript: Path) -> dict:
        events = transcript_lines(transcript)
        usage = [e["usage"] for e in events if e.get("type") == "turn.completed"]
        failed = [e for e in events if e.get("type") in ("turn.failed", "error")]
        status = "success" if usage and not failed else f"failed: {json.dumps(failed[-1])[:300]}" if failed else "no result"
        return {
            "status": status,
            "cost_usd": None,
            "input_tokens": sum(u.get("input_tokens", 0) for u in usage),
            "cached_input_tokens": sum(u.get("cached_input_tokens", 0) for u in usage),
            "output_tokens": sum(u.get("output_tokens", 0) for u in usage),
        }

    def finish(self, launch: Launch) -> None:
        """Codex may refresh its login during a Run. If it replaced the symlink with a newer file, carry that back, so
        the user's own Codex keeps a valid refresh token; then take the credentials out of the run folder."""
        auth = launch.run_dir / "codex-home" / "auth.json"
        if auth.is_file() and not auth.is_symlink() and auth.stat().st_mtime > self.AUTH.stat().st_mtime:
            shutil.copy2(auth, self.AUTH)
        auth.unlink(missing_ok=True)


EDITORS: dict[str, Editor] = {"claude": Claude(), "codex": Codex()}


def codex_profile(run_dir: Path) -> str:
    """Hide the repo, the Hugging Face cache, Claude's and Codex's own folders, and every other Run; allow writes only to
    this Run's Codex home and work folder, the user's temporary folders and Codex's login file."""
    home, run = str(Path.home()), str(run_dir.resolve())
    user_temp = str(Path(tempfile.gettempdir()).resolve().parent)
    auth = re.escape(str(Codex.AUTH))
    return f"""(version 1)
(allow default)
(deny file-read* (subpath "{REPO}") (subpath "{home}/.cache/huggingface") (subpath "{home}/.claude") (subpath "{home}/.codex") (subpath "{RUNS_ROOT}"))
(allow file-read* (subpath "{run}/codex-home") (subpath "{run}/editor") (regex #"^{auth}"))
(allow file-read-metadata)
(deny file-write*)
(allow file-write* (subpath "{run}/codex-home") (subpath "{run}/editor") (subpath "{user_temp}") (regex #"^{auth}")
  (literal "/dev/null") (literal "/dev/tty") (literal "/dev/dtracehelper"))
"""


def prompt(experiment: Path, config: RunConfig) -> str:
    fields = {**asdict(config), "sandbox_minutes": f"{config.sandbox_timeout / 60:.0f}"}
    return Template((experiment / "editor.md").read_text()).substitute(fields)


def next_name(folder: Path, prefix: str) -> str:
    taken = {p.name for p in folder.iterdir()} if folder.is_dir() else set()
    return next(f"{prefix}{n:02d}" for n in range(1, 100) if f"{prefix}{n:02d}" not in taken)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextmanager
def serving(run_dir: Path) -> Iterator[str]:
    """Run the trusted MCP server for this Run and yield its URL. The path carries a secret, so only the Editor that
    is handed the URL can reach the Session."""
    port, path = free_port(), f"/mcp/{secrets.token_urlsafe(16)}"
    with open(run_dir / "server.log", "w") as log:
        server = subprocess.Popen(
            [sys.executable, "-m", "weightspace.server", str(run_dir), str(port), path],
            cwd=run_dir, env=clean_env(), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        )  # fmt: skip
    try:
        deadline = time.monotonic() + SERVER_START_SECONDS
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"MCP server exited at start; see {run_dir / 'server.log'}")
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError("MCP server did not start") from None
                time.sleep(0.2)
        yield f"http://127.0.0.1:{port}{path}"
    finally:
        server.terminate()
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()


def git_state() -> dict:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True).stdout.strip())
    return {"commit": head, "dirty": dirty}


def launch(experiment: Path, plan: Plan, editor: str, dry: bool, text: str | None = None) -> tuple[Path, dict]:
    """Run one Editor to completion and return the run folder and its outcome. `text` replaces the prompt (for tests)."""
    if editor not in plan.editors:
        raise KeyError(f"{experiment.name} has no {editor!r} Editor in run.json; it has {sorted(plan.editors)}")
    config = replace(plan.session, **DRY) if dry else plan.session
    name = next_name(RUNS_ROOT / experiment.name, f"dry-{editor}-" if dry else f"{editor}-")
    run_dir = RUNS_ROOT / experiment.name / name
    (run_dir / "editor").mkdir(parents=True)
    config.save(run_dir / "config.json")
    text = text if text is not None else prompt(experiment, config)
    (run_dir / "prompt.md").write_text(text)
    settings = plan.editors[editor]
    meta = {"experiment": experiment.name, "run": name, "editor": editor, **asdict(settings), "dry": dry, "started": time.time(), **git_state()}
    (run_dir / "launch.json").write_text(json.dumps(meta, indent=2))
    with serving(run_dir) as url:
        run = Launch(settings, text, run_dir, url)
        argv, env = EDITORS[editor].command(run)
        wall, awake = time.time(), time.monotonic()
        try:
            with open(run_dir / "transcript.jsonl", "w") as transcript, open(run_dir / "editor.log", "w") as log:
                code = subprocess.run([*STAY_AWAKE, *argv], cwd=run.workdir, env=env, stdin=subprocess.DEVNULL, stdout=transcript, stderr=log).returncode
        finally:
            EDITORS[editor].finish(run)
        wall, awake = round(time.time() - wall), round(time.monotonic() - awake)
    outcome = {"editor": editor, **asdict(settings), "exit_code": code, "wall_seconds": wall, "asleep_seconds": wall - awake, **EDITORS[editor].outcome(run_dir / "transcript.jsonl")}
    (run_dir / "outcome.json").write_text(json.dumps(outcome, indent=2))
    if not dry:
        keep(run_dir, experiment / "runs" / name)
    return run_dir, outcome


def not_ready(free_bytes: int, on_battery: bool) -> list[str]:
    """Why this Mac should not start a real Run now."""
    problems = []
    if on_battery:
        problems.append("the Mac is on battery; plug it in, because it sleeps when the lid closes")
    if free_bytes < MIN_FREE_BYTES:
        problems.append(f"only {free_bytes / 1e9:.1f} GB of disk is free; a Run needs {MIN_FREE_BYTES / 1e9:.0f} GB")
    return problems


def machine_state() -> tuple[int, bool]:
    RUNS_ROOT.mkdir(exist_ok=True)
    power = subprocess.run(["/usr/bin/pmset", "-g", "batt"], capture_output=True, text=True).stdout
    return shutil.disk_usage(RUNS_ROOT).free, "Battery Power" in power


def too_big_for_git(folder: str, names: list[str]) -> list[str]:
    return [n for n in names if n.endswith(".patch.npz") and (Path(folder) / n).stat().st_size > GIT_FILE_LIMIT]


def keep(run_dir: Path, record: Path) -> None:
    record.mkdir(parents=True)
    for part in RECORD:
        source = run_dir / part
        if source.is_dir():
            shutil.copytree(source, record / part, ignore=too_big_for_git)
            for name in too_big_for_git(str(source), [p.name for p in source.iterdir()]):
                big = source / name
                with open(big, "rb") as data:
                    digest = hashlib.file_digest(data, "sha256").hexdigest()
                pointer = {"path": str(big), "bytes": big.stat().st_size, "sha256": digest}
                (record / part / name.replace(".npz", ".json")).write_text(json.dumps(pointer, indent=2) + "\n")
        elif source.is_file():
            (record / part).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source, record / part)
