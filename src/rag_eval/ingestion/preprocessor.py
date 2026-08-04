"""Text preprocessing and normalisation utilities."""

from __future__ import annotations

import re
import unicodedata


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace and strip leading/trailing space.

    Preserves single newlines as paragraph separators while collapsing
    double-newlines to a single blank line.
    """
    # Normalise Unicode whitespace characters
    text = unicodedata.normalize("NFKC", text)
    # Collapse horizontal whitespace (spaces, tabs) within lines
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    # Collapse 3+ blank lines into 2
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return cleaned.strip()


def remove_boilerplate(text: str) -> str:
    """Remove common document boilerplate (page numbers, headers, footers).

    Heuristic: strip lines that are:
    - pure numeric page markers
    - short non-sentence footer/header lines
    - explicit signature/footer markers
    """
    lines = text.splitlines()
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append(line)
            continue
        # Lone page numbers
        if re.fullmatch(r"\d{1,4}", stripped):
            continue
        # Short footer/header lines only
        if len(stripped) <= 80 and re.search(
            r"^(?:page\s*\d+(?:\s*of\s*\d+)?|all rights reserved|©|\*{3,}|\[\*{3}\].*|signature page|confidential(?: information)?)[\s.:-]*$",
            stripped,
            re.I,
        ):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def clean_text(text: str, *, remove_boilerplate_: bool = True) -> str:
    """Apply full preprocessing pipeline to *text*.

    Steps applied in order:

    1. Unicode normalisation (NFKC)
    2. Optional boilerplate removal
    3. Whitespace normalisation

    Args:
        text: Raw input text.
        remove_boilerplate_: Whether to run :func:`remove_boilerplate`.

    Returns:
        Cleaned text string.
    """
    normalized = normalize_whitespace(text)
    if remove_boilerplate_:
        cleaned = normalize_whitespace(remove_boilerplate(normalized))
        if cleaned.strip():
            return cleaned
    return normalized
