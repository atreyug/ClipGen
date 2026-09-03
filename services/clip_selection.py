"""Helpers for validating LLM output and selecting a safe fallback clip."""

from __future__ import annotations

import json
from typing import Any, Dict, List


def parse_llm_clip_response(result: Any) -> Dict[str, Any]:
    """Return a parsed clip response from dict or JSON-string model output."""
    if isinstance(result, dict):
        parsed = result
    elif isinstance(result, str):
        value = result.strip()
        if value.startswith("```"):
            lines = value.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            value = "\n".join(lines).strip()

        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            # Some models add a short preamble despite the schema request.
            start = value.find("{")
            end = value.rfind("}")
            if start < 0 or end <= start:
                raise ValueError("Model response does not contain a JSON object")
            parsed = json.loads(value[start:end + 1])
    else:
        raise ValueError(
            f"Model response must be an object or JSON string, got {type(result).__name__}"
        )

    if not isinstance(parsed, dict) or not isinstance(parsed.get("clips"), list):
        raise ValueError("Model response is missing a valid 'clips' array")
    return parsed


def fallback_clip_suggestions(
    transcript: List[Dict[str, Any]],
    min_duration: float = 10.0,
    max_duration: float = 60.0,
) -> List[Dict[str, Any]]:
    """Select one content-dense window made only from complete transcript segments.

    This is used only when the model returns no suggestions. Short videos are
    returned in full even when they do not meet the normal ten-second target.
    """
    segments = []
    for segment in transcript:
        try:
            start = float(segment["start"])
            end = float(segment["end"])
        except (KeyError, TypeError, ValueError):
            continue
        text = str(segment.get("text", "")).strip()
        if start < 0 or end <= start or not text:
            continue
        segments.append({"start": start, "end": end, "text": text})

    if not segments:
        return []

    segments.sort(key=lambda item: item["start"])
    full_duration = segments[-1]["end"] - segments[0]["start"]
    if full_duration <= max_duration:
        return [{
            "start": segments[0]["start"],
            "end": segments[-1]["end"],
            "viral_score": 5,
            "reason": "Complete transcript fallback",
        }]

    word_prefix = [0]
    for segment in segments:
        word_prefix.append(word_prefix[-1] + len(segment["text"].split()))

    best = None
    right = 0
    for left, first in enumerate(segments):
        right = max(right, left)
        while (right + 1 < len(segments)
               and segments[right + 1]["end"] - first["start"] <= max_duration):
            right += 1

        duration = segments[right]["end"] - first["start"]
        if duration < min_duration or duration > max_duration:
            continue
        word_count = word_prefix[right + 1] - word_prefix[left]
        score = (word_count, duration)
        if best is None or score > best[0]:
            best = (score, first["start"], segments[right]["end"])

    if best is None:
        end = min(segments[-1]["end"], segments[0]["start"] + max_duration)
        return [{
            "start": segments[0]["start"],
            "end": end,
            "viral_score": 5,
            "reason": "Complete transcript fallback",
        }]

    return [{
        "start": best[1],
        "end": best[2],
        "viral_score": 5,
        "reason": "Content-rich transcript fallback",
    }]