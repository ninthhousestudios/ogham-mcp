"""Detect AI client and install hooks configuration."""

import json
import shutil
from pathlib import Path

from rich.console import Console

console = Console()


def _detect_client() -> str:
    """Detect which AI coding client is in use."""
    if (Path.home() / ".claude" / "settings.json").exists() or shutil.which("claude"):
        return "claude-code"

    if (Path.home() / ".kiro").exists() or shutil.which("kiro"):
        return "kiro"

    if (Path.home() / ".cursor").exists():
        return "cursor"

    return "generic"


def _install_claude_code():
    """Write Claude Code hooks to settings.json."""
    settings_path = Path.home() / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)

    settings = {}
    if settings_path.exists():
        with open(settings_path) as f:
            settings = json.load(f)

    hooks = settings.setdefault("hooks", {})
    ogham_hooks = {
        "SessionStart": [{"command": "ogham hooks session-start"}],
        "PostToolUse": [{"command": "ogham hooks post-tool"}],
        "PreCompact": [{"command": "ogham hooks inscribe"}],
        "PostCompact": [{"command": "ogham hooks recall"}],
    }

    for event, cmds in ogham_hooks.items():
        existing = hooks.get(event, [])
        # Don't duplicate if already installed
        existing_cmds = [c.get("command", "") for c in existing]
        for cmd in cmds:
            if cmd["command"] not in existing_cmds:
                existing.append(cmd)
        hooks[event] = existing

    settings["hooks"] = hooks
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=2)

    console.print(f"[green]Claude Code hooks installed to {settings_path}[/green]")
    console.print("  SessionStart, PostToolUse, PreCompact, PostCompact")


def _install_kiro():
    """Print Kiro hook instructions (Kiro uses UI-based config)."""
    console.print("[bold]Kiro hooks -- manual setup via Hook UI:[/bold]")
    console.print("")
    console.print("  1. Open Command Palette (Cmd+Shift+P / Ctrl+Shift+P)")
    console.print("  2. Type 'Kiro: Open Kiro Hook UI'")
    console.print("  3. Create these hooks:")
    console.print("")
    console.print("  [bold]Hook 1: Session Start[/bold]")
    console.print("    Event: User prompt submit")
    console.print("    Action: Run Command")
    console.print("    Command: ogham hooks session-start")
    console.print("")
    console.print("  [bold]Hook 2: Post Tool[/bold]")
    console.print("    Event: Post tool invocation")
    console.print("    Action: Run Command")
    console.print("    Command: ogham hooks post-tool")
    console.print("")
    console.print(
        "[dim]Kiro doesn't support compaction hooks. Session start covers context loading.[/dim]"
    )


def _install_generic():
    """Add CLAUDE.md instructions for clients without hook support."""
    claude_md = Path.cwd() / "CLAUDE.md"

    hook_section = """
## Ogham Memory Hooks

At the start of every session, search Ogham for relevant context:
- Call `hybrid_search` with a query describing the current project
- Review the top results for decisions, gotchas, and patterns

When you finish significant work, save learnings:
- Use `store_memory` for decisions, gotchas, and architectural patterns
- Tag with type:decision, type:gotcha, type:pattern
"""

    if claude_md.exists():
        content = claude_md.read_text()
        if "Ogham Memory Hooks" in content:
            console.print("[yellow]CLAUDE.md already has Ogham hook instructions[/yellow]")
            return
        with open(claude_md, "a") as f:
            f.write(hook_section)
    else:
        claude_md.write_text(hook_section)

    console.print("[green]CLAUDE.md updated with Ogham hook instructions[/green]")
    console.print("  Works with Codex, Cursor, OpenCode, and any MCP client")


def install_hooks():
    """Detect client and install appropriate hooks."""
    client = _detect_client()
    console.print(f"Detected client: [bold]{client}[/bold]")
    console.print("")

    match client:
        case "claude-code":
            _install_claude_code()
        case "kiro":
            _install_kiro()
        case _:
            _install_generic()
            if client != "generic":
                console.print(
                    f"\n[dim]{client} doesn't support hooks natively."
                    " CLAUDE.md instructions added as fallback.[/dim]"
                )
