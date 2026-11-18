import argparse
import re
import sys
from datetime import date
from pathlib import Path

from weightspace import card

SLUG = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
NUMBERED = re.compile(r"(\d{4})-")


class CliError(Exception):
    pass


def experiments_dir() -> Path:
    root = Path.cwd() / "experiments"
    if not (root / "_template" / "card.md").is_file():
        raise CliError("run ws from the repo root (experiments/_template/card.md not found)")
    return root


def template() -> str:
    return (experiments_dir() / "_template" / "card.md").read_text()


def experiments() -> list[Path]:
    return sorted(p for p in experiments_dir().iterdir() if p.is_dir() and NUMBERED.match(p.name))


def find(ref: str) -> Path:
    for folder in experiments():
        number = NUMBERED.match(folder.name).group(1)
        if folder.name == ref or (ref.isdigit() and number == f"{int(ref):04d}"):
            return folder
    raise CliError(f"no experiment matches {ref!r}")


def new(slug: str) -> Path:
    if not SLUG.fullmatch(slug):
        raise CliError("slug must be lowercase words joined by hyphens, like loop-repair")
    number = max((int(NUMBERED.match(p.name).group(1)) for p in experiments()), default=0) + 1
    folder = experiments_dir() / f"{number:04d}-{slug}"
    (folder / "runs").mkdir(parents=True)
    title = f"# {number:04d}: {slug.replace('-', ' ').capitalize()}"
    (folder / "card.md").write_text(template().replace("# NNNN: Title", title, 1))
    return folder / "card.md"


def freeze(ref: str) -> Path:
    path = find(ref) / "card.md"
    text = path.read_text()
    if card.status(text) == "frozen":
        raise CliError(f"{path.parent.name} is already frozen; append to Amendments instead")
    missing = card.unfilled(text, template())
    if missing:
        raise CliError(f"{path.parent.name} still has unfilled sections: {', '.join(missing)}")
    path.write_text(card.frozen(text))
    return path


def listing() -> list[str]:
    rows = []
    for folder in experiments():
        text = (folder / "card.md").read_text()
        runs = sum(1 for _ in (folder / "runs").iterdir()) if (folder / "runs").is_dir() else 0
        todo = len(card.unfilled(text, template()))
        rows.append(f"{folder.name}  {card.status(text)}  runs={runs}  unfilled={todo}")
    return rows


def run_survey(args: argparse.Namespace) -> int:
    from weightspace import survey
    from weightspace.tasks import TASKS

    names = args.tasks or list(TASKS)
    unknown = set(names) - set(TASKS)
    if unknown:
        raise CliError(f"unknown tasks: {sorted(unknown)}; choose from {sorted(TASKS)}")
    modes = {"off": [False], "on": [True], "both": [False, True]}[args.thinking]
    stem = "-".join([date.today().isoformat(), Path(args.model).name, *(args.tasks or []), args.thinking, "sampled" if args.sampled else "greedy"])
    rows = survey.survey(args.model, names, args.n, modes, Path("surveys/raw") / f"{stem}.jsonl", args.batch_size, args.sampled)
    path = Path("surveys") / f"{stem}.md"
    path.write_text(survey.report(rows, args.model, args.n, args.sampled))
    print(f"wrote {path}")
    return 0


def run_editor(args: argparse.Namespace) -> int:
    from dataclasses import replace

    from weightspace.run import Plan, launch, machine_state, not_ready

    folder = find(args.experiment)
    if card.status((folder / "card.md").read_text()) != "frozen" and not args.dry_run:
        raise CliError(f"{folder.name} is a draft; freeze and commit its card before Run 1")
    if not args.dry_run and (problems := not_ready(*machine_state())):
        raise CliError("not starting the Run: " + "; ".join(problems))
    plan = Plan.load(folder / "run.json")
    if args.editor not in plan.editors:
        raise CliError(f"{folder.name} has no {args.editor!r} Editor in run.json; it has {sorted(plan.editors)}")
    if args.budget_usd is not None:
        plan = replace(plan, editors={**plan.editors, args.editor: replace(plan.editors[args.editor], spend_cap_usd=args.budget_usd)})
    run_dir, outcome = launch(folder, plan, args.editor, args.dry_run)
    cost = "" if outcome["cost_usd"] is None else f", ${outcome['cost_usd']:.2f}"
    print(f"run folder {run_dir}")
    print(
        f"{args.editor} {outcome['status']}{cost}, {outcome['input_tokens']:,} input tokens "
        f"({outcome['cached_input_tokens']:,} cached), {outcome['output_tokens']:,} output, {outcome['wall_seconds']}s"
    )
    if outcome["asleep_seconds"] > 60:
        print(f"warning: the Mac slept for {outcome['asleep_seconds']}s of this Run; keep it on power with the lid open")
    return 0 if outcome["status"] == "success" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ws", description="Run weightspace experiments.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("new", help="start an experiment from the card template").add_argument("slug")
    commands.add_parser("freeze", help="lock a filled card before Run 1").add_argument("experiment")
    commands.add_parser("list", help="show every experiment and its status")
    survey_parser = commands.add_parser("survey", help="score the Baseline on candidate tasks")
    survey_parser.add_argument("--model", default="models/Qwen3.5-0.8B")
    survey_parser.add_argument("--n", type=int, default=40, help="examples per task")
    survey_parser.add_argument("--tasks", nargs="+", default=None, help="task names (default: all)")
    survey_parser.add_argument("--thinking", choices=["off", "on", "both"], default="both")
    survey_parser.add_argument("--batch-size", type=int, default=4, help="larger is faster but swaps on 16 GB")
    survey_parser.add_argument("--sampled", action="store_true", help="use Qwen's recommended sampling instead of greedy")
    run_parser = commands.add_parser("run", help="start a Run: a headless Editor with only the sandbox and evaluate")
    run_parser.add_argument("experiment")
    run_parser.add_argument("--editor", choices=["claude", "codex"], default="claude")
    run_parser.add_argument("--dry-run", action="store_true", help="2 turns, 1 tiny evaluation, kept outside the experiment")
    run_parser.add_argument("--budget-usd", type=float, default=None, help="override the Editor's Spend cap (Claude only)")
    args = parser.parse_args(argv)
    try:
        if args.command == "survey":
            return run_survey(args)
        if args.command == "run":
            return run_editor(args)
        if args.command == "new":
            print(f"created {new(args.slug)}")
        elif args.command == "freeze":
            print(f"froze {freeze(args.experiment)}. Commit it before Run 1.")
        else:
            print("\n".join(listing()) or "no experiments yet")
    except CliError as error:
        print(f"ws: {error}", file=sys.stderr)
        return 1
    return 0
