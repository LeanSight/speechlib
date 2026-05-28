"""
Fuzzy speaker name → voice library slug matching.

Bridges human names from --speakers CLI flag (e.g., "Nicolás Loira") to
voice library folder slugs (e.g., "alicanto-nicolas-loira") using a
three-tier cascade: exact normalized → surname-weighted prefix → fuzzy.

Pure domain function: no I/O, no side effects.
"""

import unicodedata
from dataclasses import dataclass
from typing import Literal

from rapidfuzz import fuzz


@dataclass(frozen=True)
class MatchResult:
    slug: str
    confidence: Literal["high", "low"]
    tier: str


def _normalize(s: str) -> str:
    stripped = "".join(
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )
    return stripped.lower()


def _tokenize(s: str) -> list[str]:
    return _normalize(s).replace("-", " ").split()


def _prefix_match(a: str, b: str) -> bool:
    return a.startswith(b) or b.startswith(a)


def match_speaker_with_confidence(
    name: str,
    library_display_names: dict[str, str],
) -> MatchResult | None:
    """Match a human speaker name to a voice library slug, with confidence.

    Returns MatchResult with confidence="high" (full name match) or "low"
    (single-token, nickname, or fuzzy), or None if no match.
    """
    if not name or not name.strip():
        return None

    input_tokens = _tokenize(name)
    is_full_name = len(input_tokens) >= 2

    result = _match_display_name(name, library_display_names)
    if result is not None:
        return MatchResult(slug=result, confidence="high", tier="display_name")

    result = _match_surname_weighted(name, library_display_names)
    if result is not None:
        confidence = "high" if is_full_name else "low"
        return MatchResult(slug=result, confidence=confidence, tier="surname_prefix")

    result = _match_fuzzy(name, library_display_names)
    if result is not None:
        return MatchResult(slug=result, confidence="high", tier="fuzzy")

    return None


def match_speaker_to_library(
    name: str,
    library_display_names: dict[str, str],
) -> str | None:
    """Match a human speaker name to a voice library slug.

    Convenience wrapper that returns just the slug (or None).
    """
    result = match_speaker_with_confidence(name, library_display_names)
    return result.slug if result is not None else None


def _match_display_name(
    name: str, library: dict[str, str]
) -> str | None:
    name_norm = _normalize(name)
    for slug, display_name in library.items():
        if _normalize(display_name) == name_norm:
            return slug
    return None


def _match_surname_weighted(
    name: str, library: dict[str, str]
) -> str | None:
    input_tokens = _tokenize(name)

    candidates = []
    for slug, display_name in library.items():
        lib_tokens = list(set(_tokenize(slug) + _tokenize(display_name)))

        if len(input_tokens) == 1:
            token = input_tokens[0]
            if any(_prefix_match(lt, token) for lt in lib_tokens):
                candidates.append(slug)
        else:
            surname = input_tokens[-1]
            first_tokens = input_tokens[:-1]

            surname_matched = any(
                _prefix_match(lt, surname) for lt in lib_tokens
            )
            if not surname_matched:
                continue

            first_matched = sum(
                1
                for ft in first_tokens
                if any(_prefix_match(lt, ft) for lt in lib_tokens)
            )
            if first_matched >= 1:
                candidates.append(slug)

    if len(candidates) == 1:
        return candidates[0]
    return None


def _match_fuzzy(
    name: str,
    library: dict[str, str],
    threshold: int = 80,
    min_margin: int = 15,
) -> str | None:
    input_tokens = _tokenize(name)
    if len(input_tokens) < 2:
        return None

    name_norm = _normalize(name)
    scores: list[tuple[float, str]] = []
    for slug, display_name in library.items():
        s = max(
            fuzz.token_sort_ratio(name_norm, _normalize(display_name)),
            fuzz.token_sort_ratio(name_norm, slug.replace("-", " ")),
        )
        scores.append((s, slug))
    scores.sort(reverse=True)

    if not scores or scores[0][0] < threshold:
        return None
    if len(scores) >= 2 and scores[0][0] - scores[1][0] < min_margin:
        return None
    return scores[0][1]
