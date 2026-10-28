from __future__ import annotations

import re
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"


class VersionedPrompt:
    """Encapsulates a loaded prompt template with its declared version tag."""

    def __init__(self, name: str, version: str, content: str):
        self.name = name
        self.version = version
        self.content = content


_PROMPT_CACHE: dict[str, VersionedPrompt] = {}


def load_prompt(prompt_name: str) -> VersionedPrompt:
    """
    Loads a versioned prompt from ai/agents/prompts/<prompt_name>.md.
    Parses version metadata tag (<!-- version: X.Y.Z -->).
    """
    if prompt_name in _PROMPT_CACHE:
        return _PROMPT_CACHE[prompt_name]

    file_path = PROMPTS_DIR / f"{prompt_name}.md"
    if not file_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {file_path}")

    raw_text = file_path.read_text(encoding="utf-8").strip()

    # Extract version tag if present
    match = re.search(r"<!--\s*version:\s*([^\s>]+)\s*-->", raw_text)
    version = match.group(1) if match else "1.0.0"

    # Strip version header comment from prompt text
    content = re.sub(r"<!--\s*version:\s*.*?-->", "", raw_text).strip()

    prompt_obj = VersionedPrompt(name=prompt_name, version=version, content=content)
    _PROMPT_CACHE[prompt_name] = prompt_obj
    return prompt_obj
