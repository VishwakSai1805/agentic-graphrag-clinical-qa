"""Versioned prompt loader.

Prompts live as files in `src/llm/prompts/<name>.<version>.md` -- never inline
strings -- so the paper can print the exact text a run used and
`configs/*.yaml: agents.prompt_versions` pins which version each agent used
(agents-engineer.md). `.format(**kwargs)` fills the `{placeholders}`.

Phase 3's `src/eval/baselines.py` predates this and still carries inline
prompts; moving them here is tracked Phase 6 debt (docs/02_ROADMAP.md).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def load_prompt(name: str, version: str) -> str:
    """Return the raw template text for `<name>.<version>.md`."""
    path = PROMPTS_DIR / f"{name}.{version}.md"
    if not path.exists():
        available = sorted(p.name for p in PROMPTS_DIR.glob(f"{name}.*.md"))
        raise FileNotFoundError(f"no prompt {path.name}; have: {available}")
    return path.read_text(encoding="utf-8")


def prompt_sha(name: str, version: str) -> str:
    """Short content hash of a prompt file, for run provenance."""
    return hashlib.sha256(load_prompt(name, version).encode()).hexdigest()[:12]
