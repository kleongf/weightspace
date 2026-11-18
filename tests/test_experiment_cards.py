from pathlib import Path

import pytest

from weightspace import card

EXPERIMENTS = Path(__file__).parent.parent / "experiments"
TEMPLATE = (EXPERIMENTS / "_template" / "card.md").read_text()
CARDS = sorted(p / "card.md" for p in EXPERIMENTS.iterdir() if p.is_dir() and p.name != "_template")


@pytest.mark.parametrize("path", CARDS, ids=lambda p: p.parent.name)
def test_card_has_every_template_section(path: Path):
    missing = set(card.sections(TEMPLATE)) - set(card.sections(path.read_text()))
    assert not missing, f"{path.parent.name} card is missing sections: {sorted(missing)}"


@pytest.mark.parametrize("path", CARDS, ids=lambda p: p.parent.name)
def test_experiment_with_runs_has_frozen_card(path: Path):
    runs = path.parent / "runs"
    if not runs.is_dir() or not any(runs.iterdir()):
        pytest.skip("no runs yet")
    assert card.status(path.read_text()) == "frozen", f"{path.parent.name} has runs but its card is not frozen"


@pytest.mark.parametrize("path", CARDS, ids=lambda p: p.parent.name)
def test_frozen_card_is_filled(path: Path):
    text = path.read_text()
    if card.status(text) != "frozen":
        pytest.skip("draft")
    assert not card.unfilled(text, TEMPLATE), f"{path.parent.name} is frozen with unfilled sections"


@pytest.mark.parametrize("path", CARDS, ids=lambda p: p.parent.name)
def test_frozen_experiment_can_launch_an_editor(path: Path):
    from dataclasses import replace

    from weightspace.run import DRY, Plan, prompt

    if card.status(path.read_text()) != "frozen":
        pytest.skip("draft")
    plan = Plan.load(path.parent / "run.json")
    for config in (plan.session, replace(plan.session, **DRY)):
        text = prompt(path.parent, config)
        assert "$" not in text and str(config.max_tokens) in text
    assert plan.session.eval_budget <= 100 and plan.spend_cap_usd > 0
