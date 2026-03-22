"""CLI sub-commands for ogham hooks."""

import json
import sys

import typer

hooks_app = typer.Typer(name="hooks", help="Lifecycle hooks for AI coding clients.")


def _read_stdin() -> dict:
    """Read hook input JSON from stdin."""
    try:
        if sys.stdin.isatty():
            return {}
        raw = sys.stdin.read()
        return json.loads(raw) if raw.strip() else {}
    except (json.JSONDecodeError, Exception):
        return {}


@hooks_app.command(name="session-start")
def session_start_cmd(
    profile: str = typer.Option("work", help="Memory profile"),
):
    """Inject relevant memories at session start. Output goes to stdout."""
    from ogham.hooks import session_start

    data = _read_stdin()
    cwd = data.get("cwd", ".")
    output = session_start(cwd=cwd, profile=profile)
    if output:
        typer.echo(output)


@hooks_app.command(name="post-tool")
def post_tool_cmd(
    profile: str = typer.Option("work", help="Memory profile"),
):
    """Capture tool execution as a memory. Reads hook JSON from stdin."""
    from ogham.hooks import post_tool

    data = _read_stdin()
    if data:
        post_tool(data, profile=profile)


@hooks_app.command(name="inscribe")
def inscribe_cmd(
    profile: str = typer.Option("work", help="Memory profile"),
):
    """Inscribe session context to Ogham before compaction."""
    from ogham.hooks import pre_compact

    data = _read_stdin()
    pre_compact(
        session_id=data.get("session_id", "unknown"),
        cwd=data.get("cwd", "."),
        profile=profile,
    )


@hooks_app.command(name="recall")
def recall_cmd(
    profile: str = typer.Option("work", help="Memory profile"),
):
    """Recall context from Ogham after compaction. Output goes to stdout."""
    from ogham.hooks import post_compact

    data = _read_stdin()
    cwd = data.get("cwd", ".")
    output = post_compact(cwd=cwd, profile=profile)
    if output:
        typer.echo(output)


@hooks_app.command(name="install")
def install_cmd():
    """Detect AI client and install hooks configuration."""
    from ogham.hooks_install import install_hooks

    install_hooks()
