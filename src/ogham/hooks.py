"""Lifecycle hooks for AI coding clients.

Called by `ogham hooks <event>` or shell wrappers. Each function reads
context from the Ogham database and either outputs markdown (for context
injection) or stores a memory (for capture).

Supported clients: Claude Code (native hooks), Kiro (Hook UI),
Codex/Cursor/OpenCode (CLAUDE.md fallback).
"""

import logging
import os
import re
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Tools we never capture (prevent infinite loops)
_SKIP_PREFIXES = ("mcp__ogham__", "ogham_", "store_memory", "hybrid_search")

# Tools that are routine -- only capture if they contain signal keywords
_ROUTINE_TOOLS = frozenset({"Read", "Glob", "Grep", "Bash", "ListDir"})

# Signal keywords that make a routine tool worth capturing
_SIGNAL_KEYWORDS = frozenset(
    {
        # Errors and debugging
        "error",
        "fail",
        "fix",
        "bug",
        "broke",
        "crash",
        "exception",
        "traceback",
        "stacktrace",
        "segfault",
        "panic",
        # Decisions and changes
        "decided",
        "chose",
        "switch",
        "migrate",
        "replace",
        "refactor",
        "deprecated",
        "removed",
        "added",
        "changed",
        # Infrastructure
        "config",
        "deploy",
        "release",
        "install",
        "upgrade",
        "rollback",
        "permission",
        "denied",
        "timeout",
        "refused",
        "certificate",
        # DevOps
        "docker",
        "railway",
        "neon",
        "supabase",
        "vercel",
        "cloudflare",
        "terraform",
        "kubernetes",
        "k8s",
        "helm",
        # Testing
        "test",
        "pytest",
        "jest",
        "passed",
        "failed",
        "coverage",
        # Security
        "secret",
        "credential",
        "auth",
        "token",
        "vulnerability",
        "cve",
        # Database
        "migration",
        "schema",
        "index",
        "vacuum",
        "replication",
        # Workarounds
        "todo",
        "hack",
        "workaround",
        "gotcha",
        "caveat",
        "warning",
        # Package management
        "pip install",
        "npm install",
        "uv add",
        "go get",
        "cargo add",
    }
)

# Noise commands we never capture
_NOISE_COMMANDS = frozenset(
    {
        "ls",
        "pwd",
        "cd",
        "cat",
        "head",
        "tail",
        "wc",
        "echo",
        "date",
        "whoami",
        "which",
        "type",
        "clear",
        "history",
    }
)

# Git subcommands worth capturing (commits, pushes, merges -- not maintenance)
_GIT_SIGNAL = frozenset(
    {
        "commit",
        "push",
        "merge",
        "rebase",
        "tag",
        "release",
        "reset",
        "revert",
        "cherry-pick",
    }
)
# Git subcommands that are noise
_GIT_NOISE = frozenset(
    {
        "add",
        "status",
        "diff",
        "log",
        "show",
        "branch",
        "checkout",
        "switch",
        "fetch",
        "pull",
        "stash",
        "clean",
        "gc",
        "remote",
        "config",
    }
)

# Patterns that look like secrets -- mask these before storing.
# Two-tier approach:
# 1. _SECRET_PATTERNS: matches KEY=VALUE patterns (api_key=sk-proj-...)
# 2. _BARE_SECRET_PATTERNS: matches bare tokens without assignment (ghp_..., AKIA...)
# 3. _URL_CREDENTIALS: matches user:pass@host in URLs
# 4. _ENV_SECRET_KEYS: generic KEY=value matching for common env var names

_SECRET_PATTERNS = re.compile(
    r"(?i)"
    # Generic key=value patterns
    r"(?:api[_-]?key|secret[_-]?key|access[_-]?key|access[_-]?token|auth[_-]?token"
    r"|password|passwd|bearer|token"
    # Cloud provider prefixes
    r"|sk[_-]live|sk[_-]proj|pk[_-]live|sk[_-]test|pk[_-]test"
    # Service-specific prefixes
    r"|ghp_|gho_|github_pat_|glpat-|xoxb-|xoxp-|whsec_"
    r"|sb_secret_|ogham_live_"
    r"|SG\.[A-Za-z0-9_-]{20}"  # SendGrid
    r"|npm_[A-Za-z0-9]{20}"  # NPM
    r"|pypi-[A-Za-z0-9]{20}"  # PyPI
    # Voyage/Neon/custom
    r"|pa-[A-Za-z0-9_-]{20}"
    r"|npg_[A-Za-z0-9]{10}"
    # AWS
    r"|AKIA[A-Z0-9]{16}"
    # JWT
    r"|eyJ[A-Za-z0-9_-]{20,})"
    r"[=:\s]+\s*['\"]?([A-Za-z0-9_\-./+=]{8,})['\"]?"
)

# Bare tokens that don't need a KEY= prefix to be recognised
_BARE_SECRET_PATTERNS = re.compile(
    r"(?:"
    r"ghp_[A-Za-z0-9]{36}"  # GitHub PAT
    r"|gho_[A-Za-z0-9]{36}"  # GitHub OAuth
    r"|github_pat_[A-Za-z0-9_]{20,}"  # GitHub fine-grained PAT
    r"|glpat-[A-Za-z0-9\-]{20,}"  # GitLab PAT
    r"|AKIA[A-Z0-9]{16}"  # AWS access key
    r"|SG\.[A-Za-z0-9_\-]{20,}"  # SendGrid
    r"|xox[bpars]-[A-Za-z0-9\-]{10,}"  # Slack tokens
    r"|sk-ant-[A-Za-z0-9\-]{20,}"  # Anthropic
    r"|sk-proj-[A-Za-z0-9\-]{20,}"  # OpenAI project
    r"|sk-[A-Za-z0-9]{40,}"  # OpenAI legacy
    r"|npm_[A-Za-z0-9]{36}"  # NPM
    r"|pypi-[A-Za-z0-9]{20,}"  # PyPI
    r"|whsec_[A-Za-z0-9]{20,}"  # Webhook secret (Clerk/Svix)
    r"|ogham_live_[A-Za-z0-9_\-]{20,}"  # Ogham API key
    r"|\d{8,12}:[A-Za-z0-9_-]{35}"  # Telegram bot token
    r"|[A-Za-z0-9]{24}\.[A-Za-z0-9_-]{6}\.[A-Za-z0-9_-]{27}"  # Discord bot token
    r")"
)

# Basic auth in URLs: user:pass@host
_URL_CREDENTIALS = re.compile(r"://([^:]+):([^@]{3,})@")

_ENV_SECRET_KEYS = frozenset(
    {
        "api_key",
        "secret_key",
        "access_key",
        "access_token",
        "auth_token",
        "password",
        "passwd",
        "bearer",
        "private_key",
        "database_url",
        "connection_string",
        "dsn",
        "redis_url",
        "valkey_url",
        "mongodb_uri",
        "encryption_key",
        "signing_key",
        "webhook_secret",
    }
)

# High-value tool input fields to extract for summaries
_SUMMARY_FIELDS = ("command", "content", "query", "file_path", "url", "message")


def _mask_secrets(text: str) -> str:
    """Replace anything that looks like a secret with a masked placeholder.

    Three detection layers:
    1. KEY=value patterns (api_key=sk-proj-...)
    2. Bare tokens without assignment (ghp_..., AKIA..., sk-ant-...)
    3. URL credentials (user:pass@host)
    4. Generic env var names (password=, database_url=)

    Captures the event ("set API key for Stripe") but never the value.
    """
    # Layer 1: KEY=value patterns
    masked = _SECRET_PATTERNS.sub(
        lambda m: m.group(0)[: m.start(1) - m.start(0)] + "***MASKED***",
        text,
    )
    # Layer 2: Bare tokens (no KEY= prefix needed)
    masked = _BARE_SECRET_PATTERNS.sub("***MASKED***", masked)
    # Layer 3: URL credentials (user:pass@host)
    masked = _URL_CREDENTIALS.sub("://***MASKED***:***MASKED***@", masked)
    # Layer 4: Generic env var names
    for key in _ENV_SECRET_KEYS:
        pattern = re.compile(rf"(?i){re.escape(key)}\s*[=:]\s*['\"]?([^\s'\"]+)['\"]?")
        masked = pattern.sub(
            lambda m: m.group(0)[: m.start(1) - m.start(0)] + "***MASKED***",
            masked,
        )
    return masked


def session_start(cwd: str, profile: str = "work", limit: int = 8) -> str:
    """Return markdown context for session injection.

    Searches for memories relevant to the current working directory.
    """
    from ogham.database import hybrid_search_memories
    from ogham.embeddings import generate_embedding

    project_name = os.path.basename(cwd)
    query = f"project context for {project_name}"

    try:
        embedding = generate_embedding(query)
        results = hybrid_search_memories(
            query_text=query,
            query_embedding=embedding,
            profile=profile,
            limit=limit,
        )
    except Exception:
        logger.debug("session_start: search failed, returning empty")
        return ""

    if not results:
        return ""

    lines = ["## Session Context", ""]
    for r in results:
        content = r.get("content", "")[:200]
        tags = [t for t in r.get("tags", []) if t.startswith("type:")]
        tag_str = f" ({', '.join(tags)})" if tags else ""
        lines.append(f"- {content}{tag_str}")

    lines.append("")
    lines.append(f"*{len(results)} memories loaded for {project_name}*")
    return "\n".join(lines)


def post_tool(hook_input: dict, profile: str = "work") -> None:
    """Capture a tool execution as a memory.

    Skips Ogham's own tools to prevent infinite loops.
    """
    tool_name = hook_input.get("tool_name", "")

    if any(tool_name.startswith(p) for p in _SKIP_PREFIXES):
        return

    tool_input = hook_input.get("tool_input", {})
    cwd = hook_input.get("cwd", "")
    session_id = hook_input.get("session_id", "")

    # Extract summary from tool input
    summary = ""
    if isinstance(tool_input, dict):
        for field in _SUMMARY_FIELDS:
            if field in tool_input:
                summary = str(tool_input[field])[:200]
                break
        if not summary:
            summary = str(tool_input)[:200]
    else:
        summary = str(tool_input)[:200]

    summary_lower = summary.lower()

    # Skip pure noise commands (ls, pwd, cat, etc.)
    if tool_name == "Bash":
        parts = summary_lower.strip().split()
        cmd_word = parts[0] if parts else ""
        if cmd_word in _NOISE_COMMANDS:
            return
        # Git: only capture commits, pushes, merges -- not add, status, diff
        if cmd_word == "git" and len(parts) > 1:
            git_sub = parts[1]
            if git_sub in _GIT_NOISE:
                return
            if git_sub not in _GIT_SIGNAL:
                # Unknown git subcommand -- skip unless it has signal keywords
                if not any(kw in summary_lower for kw in _SIGNAL_KEYWORDS):
                    return

    # For routine tools, only capture if content has signal keywords
    if tool_name in _ROUTINE_TOOLS:
        if not any(kw in summary_lower for kw in _SIGNAL_KEYWORDS):
            return

    # Mask any secrets before storing
    summary = _mask_secrets(summary)

    # High-value tools always captured (Write, Edit, Agent, WebFetch, etc.)
    content = f"Tool: {tool_name}\nInput: {summary}\nDirectory: {cwd}"

    try:
        from ogham.service import store_memory_enriched

        store_memory_enriched(
            content=content,
            profile=profile,
            source="hook:post-tool",
            tags=["type:action", f"tool:{tool_name}", f"session:{session_id}"],
        )
    except Exception:
        logger.debug("post_tool: store failed, ignoring")


def pre_compact(session_id: str, cwd: str, profile: str = "work") -> None:
    """Drain session context to Ogham before compaction."""
    project_name = os.path.basename(cwd)
    timestamp = datetime.now(timezone.utc).isoformat()

    content = (
        f"Session drain before compaction.\n"
        f"Project: {project_name}\n"
        f"Directory: {cwd}\n"
        f"Session: {session_id}\n"
        f"Time: {timestamp}"
    )

    try:
        from ogham.service import store_memory_enriched

        store_memory_enriched(
            content=content,
            profile=profile,
            source="hook:pre-compact",
            tags=["type:session", f"session:{session_id}", "compaction:drain"],
        )
    except Exception:
        logger.debug("pre_compact: store failed, ignoring")


def post_compact(cwd: str, profile: str = "work", limit: int = 10) -> str:
    """Rehydrate context after compaction.

    Returns markdown with the most relevant memories for the project.
    """
    from ogham.database import hybrid_search_memories
    from ogham.embeddings import generate_embedding

    project_name = os.path.basename(cwd)
    query = f"recent work and decisions for {project_name}"

    try:
        embedding = generate_embedding(query)
        results = hybrid_search_memories(
            query_text=query,
            query_embedding=embedding,
            profile=profile,
            limit=limit,
        )
    except Exception:
        logger.debug("post_compact: search failed, returning empty")
        return ""

    if not results:
        return ""

    lines = ["## Restored Context", ""]
    for r in results:
        content = r.get("content", "")[:300]
        tags = [t for t in r.get("tags", []) if t.startswith("type:")]
        tag_str = f" ({', '.join(tags)})" if tags else ""
        lines.append(f"- {content}{tag_str}")

    lines.append("")
    lines.append(f"*{len(results)} memories restored for {project_name} after compaction*")
    return "\n".join(lines)
