import shutil
from pathlib import Path

import pytest

from weightspace import card
from weightspace.cli import main

REPO = Path(__file__).parent.parent


@pytest.fixture
def lab(tmp_path, monkeypatch):
    shutil.copytree(REPO / "experiments" / "_template", tmp_path / "experiments" / "_template")
    monkeypatch.chdir(tmp_path)
    return tmp_path / "experiments"


def fill(path: Path) -> None:
    text = path.read_text()
    for name, body in card.sections(text).items():
        if name != "Amendments":
            text = text.replace(body, f"Filled {name}.", 1)
    path.write_text(text)


def test_new_numbers_experiments_in_order(lab):
    assert main(["new", "loop-repair"]) == 0
    assert main(["new", "format-fix"]) == 0
    assert sorted(p.name for p in lab.iterdir()) == ["0001-loop-repair", "0002-format-fix", "_template"]
    assert card.status((lab / "0001-loop-repair" / "card.md").read_text()) == "draft"


def test_new_rejects_bad_slug(lab, capsys):
    assert main(["new", "Loop Repair"]) == 1
    assert "slug" in capsys.readouterr().err


def test_freeze_refuses_unfilled_card(lab, capsys):
    main(["new", "loop-repair"])
    assert main(["freeze", "1"]) == 1
    assert "Hypothesis" in capsys.readouterr().err
    assert card.status((lab / "0001-loop-repair" / "card.md").read_text()) == "draft"


def test_freeze_locks_filled_card_once(lab, capsys):
    main(["new", "loop-repair"])
    fill(lab / "0001-loop-repair" / "card.md")
    assert main(["freeze", "0001-loop-repair"]) == 0
    assert card.status((lab / "0001-loop-repair" / "card.md").read_text()) == "frozen"
    assert main(["freeze", "1"]) == 1
    assert "already frozen" in capsys.readouterr().err


def test_list_shows_status_and_progress(lab, capsys):
    main(["new", "loop-repair"])
    capsys.readouterr()
    main(["list"])
    assert "0001-loop-repair  draft  runs=0  unfilled=11" in capsys.readouterr().out
