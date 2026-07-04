from __future__ import annotations

import json
from typing import Any


def coerce_json_dict(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    if text.startswith("```"):
        stripped_lines = [
            line
            for line in text.splitlines()
            if not line.strip().startswith("```")
        ]
        text = "\n".join(stripped_lines).strip()
        if not text:
            return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        repaired = repair_json_fragment(text)
        if repaired is not None:
            try:
                payload = json.loads(repaired)
            except json.JSONDecodeError:
                payload = _extract_embedded_json(text)
        else:
            payload = _extract_embedded_json(text)
    if not isinstance(payload, dict):
        return None
    return payload


def repair_json_fragment(text: str) -> str | None:
    candidate = text.strip()
    if not candidate or "{" not in candidate:
        return None
    open_braces = candidate.count("{")
    close_braces = candidate.count("}")
    open_brackets = candidate.count("[")
    close_brackets = candidate.count("]")
    if close_brackets < open_brackets:
        candidate += "]" * (open_brackets - close_brackets)
    if close_braces < open_braces:
        candidate += "}" * (open_braces - close_braces)
    return candidate


def _extract_embedded_json(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload
