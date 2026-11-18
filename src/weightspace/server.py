"""The MCP server a Run's Editor talks to. Usage: python -m weightspace.server <run folder holding config.json>"""

import sys
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from weightspace.session import RunConfig, Session


def build(session: Session) -> MCPServer:
    config = session.config
    server = MCPServer("weightspace")

    @server.tool(
        description=(
            "Run Python in your persistent sandbox; variables persist between calls and a final expression's value is shown. "
            "`model` is the Target as an mlx module and `tokenizer` its tokenizer. "
            "`target.complete(prompts, thinking=..., max_tokens=..., sampled=...)` generates text and returns a list of "
            "Completion objects with fields text, answer (the text after </think>, or None), finished (False when cut "
            "off at max_tokens), tokens and looped. "
            "`practice` lists the Practice set as dicts with id, prompt and reference; it is also in practice.json. "
            "Edit weights in place, for example with model.update(...). Only weight values count: keep every tensor's name, "
            "shape and dtype. Gradients are banned. Files you write stay in your sandbox folder, where evals/ holds each "
            f"evaluation's Patch and full outputs. A call stops after {config.sandbox_timeout / 60:.0f} minutes."
        )
    )
    def python(code: str) -> str:
        return session.python(code)

    @server.tool(
        description=(
            f"Score the current weights on up to {config.eval_size} Practice examples, by id; the default is the first "
            f"{config.eval_size}. Each call spends one evaluation from your Eval budget of {config.eval_budget}. "
            "Returns each example's Strict and Lenient score and its output."
        )
    )
    def evaluate(example_ids: list[str] | None = None) -> str:
        return session.evaluate(example_ids)

    return server


def main() -> None:
    run_dir = Path(sys.argv[1]).resolve()
    session = Session(run_dir, RunConfig.load(run_dir / "config.json"))
    try:
        build(session).run()
    finally:
        session.close()


if __name__ == "__main__":
    main()
