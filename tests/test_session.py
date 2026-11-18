"""The sandbox, `evaluate` and the MCP server, run against the real Target with tiny settings (about 30 seconds)."""

import sys
from pathlib import Path

import anyio
import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from weightspace.patch import Patch
from weightspace.session import REPO, RunConfig, Session
from weightspace.tasks import TASKS

MODEL = REPO / "models" / "Qwen3.5-0.8B"
pytestmark = pytest.mark.skipif(not MODEL.is_dir(), reason="Target weights not downloaded")
GATE = "model.language_model.model.layers[0].mlp.gate_proj"


def config(**changes) -> RunConfig:
    settings = dict(
        task="json-extraction",
        model=str(MODEL),
        thinking=False,
        sampled=True,
        max_tokens=16,
        batch_size=2,
        practice_size=4,
        eval_size=2,
        eval_budget=2,
        turn_limit=100,
        sandbox_timeout=5,
    )
    return RunConfig(**{**settings, **changes})


@pytest.fixture(scope="module", autouse=True)
def no_disk_reserve():
    """These tests check the sandbox, not the disk guard, which has its own test. Swap can leave this Mac under 1 GB."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("weightspace.session.DISK_RESERVE", 0)
        yield


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    s = Session(tmp_path_factory.mktemp("run"), config())
    yield s
    s.close()


def refused(output: str) -> bool:
    return "PermissionError" in output or "ModuleNotFoundError" in output


@pytest.mark.parametrize(
    "code",
    [
        "import weightspace.tasks",
        f"open({str(REPO / 'src/weightspace/tasks.py')!r}).read()",
        f"open({str(REPO / 'experiments/_template/card.md')!r}).read()",
        f"import os; os.listdir({str(Path.home() / '.cache/huggingface')!r})",
    ],
    ids=["import-tasks", "read-tasks-source", "read-repo", "read-dataset-cache"],
)
def test_sandbox_cannot_reach_anything_that_yields_the_held_out_set(session, code):
    assert refused(session.python(code))


@pytest.mark.parametrize(
    "code",
    [
        f"open({str(REPO / 'tampered.txt')!r}, 'w')",
        f"open({str(Path.home() / 'tampered.txt')!r}, 'w')",
        "import socket; socket.create_connection(('1.1.1.1', 443), timeout=3)",
    ],
    ids=["write-repo", "write-home", "network"],
)
def test_sandbox_cannot_write_outside_its_folder_or_use_the_network(session, code):
    assert refused(session.python(code))


def test_sandbox_can_write_in_its_own_folder(session):
    assert session.python("open('notes.txt', 'w').write('ok')") == "2\n"


def test_gradients_are_refused(session):
    assert "banned" in session.python("mx.grad(lambda a: a.sum())(mx.ones(2))")


def test_state_persists_between_calls(session):
    session.python("kept = 41")
    assert session.python("kept + 1") == "42\n"


def test_a_cell_past_the_timeout_is_interrupted_and_state_survives(session):
    session.python("before_sleep = 'still here'")
    assert "Interrupted" in session.python("import time; time.sleep(60)")
    assert session.python("before_sleep") == "'still here'\n"


def test_evaluate_records_the_patch_and_spends_the_budget(session):
    assert "no evaluation spent" in session.evaluate(["json-practice-999"])
    session.python(f"w = {GATE}.weight\nw[0, 0] = 1.0\n{GATE}.weight = w")
    report = session.evaluate()
    assert report.startswith("Evaluation 1 of 2 (1 left)")
    assert Patch.load(session.run_dir / "evals" / "01.patch.npz").size == 1
    session.evaluate(["json-practice-2"])
    assert not (session.run_dir / "evals" / "01.patch.npz").exists(), "a tie keeps only the later Patch"
    assert (session.run_dir / "evals" / "02.patch.npz").exists()
    assert "budget of 2 is spent" in session.evaluate()


def test_a_patch_that_scores_lower_than_the_best_is_not_kept(tmp_path):
    s = Session(tmp_path, config(max_tokens=96))
    try:
        s.evaluate()
        assert s.best[1] > 0, "the Baseline must score above zero for this test to mean anything"
        s.python("n = model.language_model.model.norm\nn.weight = mx.zeros_like(n.weight)")
        assert "Patch not kept: evaluation 1 scored higher" in s.evaluate()
        assert sorted(p.name for p in (tmp_path / "evals").glob("*.patch.npz")) == ["01.patch.npz"]
        assert sorted(p.name for p in (tmp_path / "sandbox" / "evals").glob("*.patch.npz")) == ["01.patch.npz"]
    finally:
        s.close()


def test_a_patch_too_big_for_the_disk_is_refused_without_spending(tmp_path, monkeypatch):
    import weightspace.session

    monkeypatch.setattr(weightspace.session, "DISK_RESERVE", 10**15)
    s = Session(tmp_path, config())
    try:
        s.python(f"w = {GATE}.weight\nw[0, 0] = 1.0\n{GATE}.weight = w")
        assert "make a smaller edit" in s.evaluate()
        assert s.evaluations == 0 and not list(tmp_path.rglob("*.patch.npz"))
    finally:
        s.close()


def test_evaluate_refuses_a_dtype_change_without_spending(tmp_path):
    s = Session(tmp_path, config())
    try:
        s.python(f"{GATE}.weight = {GATE}.weight.astype(mx.float32)")
        assert "must keep shape" in s.evaluate()
        assert s.evaluations == 0
    finally:
        s.close()


def test_the_turn_limit_ends_the_run(tmp_path):
    s = Session(tmp_path, config(turn_limit=0))
    assert "Turn limit of 0 is reached" in s.python("1")
    assert s.worker is None


def test_held_out_prompts_never_enter_the_run_folder(session):
    session.python("text = open('practice.json').read()")
    held_out = [e.prompt for e in TASKS["json-extraction"].examples("held-out", 100)]
    files = [p for p in session.run_dir.rglob("*") if p.is_file()]
    assert files
    for path in files:
        content = path.read_bytes().decode("utf-8", errors="ignore")
        assert not any(prompt in content for prompt in held_out), path


def test_mcp_server_offers_exactly_the_two_tools(tmp_path):
    config().save(tmp_path / "config.json")

    async def tool_names() -> set[str]:
        parameters = StdioServerParameters(command=sys.executable, args=["-m", "weightspace.server", str(tmp_path)])
        async with stdio_client(parameters) as (read, write), ClientSession(read, write) as client:
            await client.initialize()
            return {tool.name for tool in (await client.list_tools()).tools}

    assert anyio.run(tool_names) == {"python", "evaluate"}
