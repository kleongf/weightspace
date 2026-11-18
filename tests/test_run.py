"""How a Run reaches its Editor: the trusted HTTP server and the sandbox around Codex. No model calls, a few seconds."""

import subprocess
from pathlib import Path

import anyio
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

import json

import weightspace.run
from weightspace.run import RUNS_ROOT, Codex, codex_profile, keep, not_ready, serving
from weightspace.session import REPO, RunConfig

HIDDEN = {
    "tasks-source": REPO / "src/weightspace/tasks.py",
    "card": REPO / "experiments/0001-json-thinking-overrun/card.md",
    "codex-config": Path.home() / ".codex/config.toml",
    "claude-settings": Path.home() / ".claude/settings.json",
}


def config() -> RunConfig:
    return RunConfig("json-extraction", str(REPO / "models/Qwen3.5-0.8B"), False, True, 16, 2, 4, 2, 2, 100, 5)


@pytest.fixture
def run_dir(tmp_path):
    folder = RUNS_ROOT / "_test" / tmp_path.name
    (folder / "editor").mkdir(parents=True)
    (folder / "codex-home").mkdir()
    yield folder
    subprocess.run(["rm", "-rf", str(folder)])


def sandboxed(run_dir: Path, *argv: str) -> int:
    return subprocess.run(["/usr/bin/sandbox-exec", "-p", codex_profile(run_dir), *argv], capture_output=True).returncode


@pytest.mark.parametrize("name", HIDDEN)
def test_codex_cannot_read_the_repo_or_other_tools_settings(run_dir, name):
    if not HIDDEN[name].is_file():
        pytest.skip(f"{HIDDEN[name]} does not exist here")
    assert sandboxed(run_dir, "/bin/cat", str(HIDDEN[name])) != 0


def test_codex_cannot_read_other_runs(run_dir):
    other = run_dir.parent / f"{run_dir.name}-other"
    other.mkdir()
    (other / "log.jsonl").write_text("secret\n")
    try:
        assert sandboxed(run_dir, "/bin/cat", str(other / "log.jsonl")) != 0
    finally:
        subprocess.run(["rm", "-rf", str(other)])


def test_codex_can_use_its_own_folders_and_login(run_dir):
    assert sandboxed(run_dir, "/usr/bin/touch", str(run_dir / "editor" / "note")) == 0
    assert sandboxed(run_dir, "/usr/bin/touch", str(run_dir / "codex-home" / "state")) == 0
    if Codex.AUTH.is_file():
        assert sandboxed(run_dir, "/bin/cat", str(Codex.AUTH)) == 0


def test_codex_cannot_write_to_the_repo(run_dir):
    assert sandboxed(run_dir, "/usr/bin/touch", str(REPO / "tampered.txt")) != 0
    assert not (REPO / "tampered.txt").exists()


def test_http_server_offers_exactly_the_two_tools(tmp_path):
    config().save(tmp_path / "config.json")

    async def tool_names(url: str) -> set[str]:
        async with streamable_http_client(url) as (read, write), ClientSession(read, write) as client:
            await client.initialize()
            return {tool.name for tool in (await client.list_tools()).tools}

    with serving(tmp_path) as url:
        assert anyio.run(tool_names, url) == {"python", "evaluate"}


def test_a_real_run_needs_power_and_disk():
    assert not_ready(10**10, on_battery=False) == []
    assert len(not_ready(10**9, on_battery=True)) == 2


def test_the_record_names_a_patch_too_big_for_git_instead_of_copying_it(tmp_path, monkeypatch):
    monkeypatch.setattr(weightspace.run, "GIT_FILE_LIMIT", 100)
    evals = tmp_path / "run" / "evals"
    evals.mkdir(parents=True)
    (evals / "01.jsonl").write_text("{}\n")
    (evals / "01.patch.npz").write_bytes(b"x" * 101)
    keep(tmp_path / "run", tmp_path / "record")
    kept = tmp_path / "record" / "evals"
    assert sorted(p.name for p in kept.iterdir()) == ["01.jsonl", "01.patch.json"]
    assert json.loads((kept / "01.patch.json").read_text())["bytes"] == 101
