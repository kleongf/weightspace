import argparse
import re
import sys
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ws", description="Run weightspace experiments.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("new", help="start an experiment from the card template").add_argument("slug")
    commands.add_parser("freeze", help="lock a filled card before Run 1").add_argument("experiment")
    commands.add_parser("list", help="show every experiment and its status")
    args = parser.parse_args(argv)
    try:
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
