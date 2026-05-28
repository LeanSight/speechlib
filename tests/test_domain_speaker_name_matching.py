"""
Domain tests for fuzzy speaker name → voice library slug matching.

The voice library uses folder slugs (e.g., "alicanto-nicolas-loira") while
the --speakers CLI flag uses human names (e.g., "Nicolás Loira"). The matching
function bridges these two representations.
"""

import pytest
from speechlib.domain.name_matching import match_speaker_to_library, MatchResult


# ── Fixtures ─────────────────────────────────────────────────────────────────

LIBRARY_DISPLAY_NAMES = {
    "agustin-villena": "Agustín Villena",
    "alicanto-nicolas-loira": "Nicolás Loira",
    "carlos-soublette": "Carlos Soublette",
    "cristian-correa": "Cristian Correa",
    "juan-pablo-traverso": "Juan Pablo Traverso",
    "manuel-olguin": "Manuel Olguín",
    "patricio-renner": "Patricio Renner",
    "paula-lapostol": "Paula Lapostol",
    "ximena-vial": "Ximena Vial",
}


# ── Exact display_name matches (with accent normalization) ───────────────────


class TestExactDisplayNameMatch:
    """
    Given  a voice library with display_name metadata
    When   the speaker name matches a display_name exactly (modulo accents/case)
    Then   the corresponding slug is returned
    """

    @pytest.mark.parametrize(
        "name, expected_slug",
        [
            ("Agustín Villena", "agustin-villena"),
            ("Agustin Villena", "agustin-villena"),
            ("agustin villena", "agustin-villena"),
            ("Nicolás Loira", "alicanto-nicolas-loira"),
            ("Nicolas Loira", "alicanto-nicolas-loira"),
            ("Carlos Soublette", "carlos-soublette"),
            ("Juan Pablo Traverso", "juan-pablo-traverso"),
            ("Ximena Vial", "ximena-vial"),
        ],
    )
    def test_matches_display_name(self, name, expected_slug):
        result = match_speaker_to_library(name, LIBRARY_DISPLAY_NAMES)
        assert result == expected_slug


# ── Nickname / prefix matches ────────────────────────────────────────────────


class TestNicknameMatch:
    """
    Given  a voice library with enrolled speakers
    When   the speaker name uses a nickname that is a prefix of a known name
    Then   the correct slug is returned
    """

    @pytest.mark.parametrize(
        "name, expected_slug",
        [
            ("Nico Loira", "alicanto-nicolas-loira"),
            ("Pato Renner", "patricio-renner"),
        ],
    )
    def test_nickname_prefix_matches(self, name, expected_slug):
        result = match_speaker_to_library(name, LIBRARY_DISPLAY_NAMES)
        assert result == expected_slug


# ── Single-token unique match ────────────────────────────────────────────────


class TestSingleTokenMatch:
    """
    Given  a voice library where a name token uniquely identifies one speaker
    When   only that token is provided
    Then   the unique match is returned
    """

    @pytest.mark.parametrize(
        "name, expected_slug",
        [
            ("Villena", "agustin-villena"),
            ("Loira", "alicanto-nicolas-loira"),
        ],
    )
    def test_unique_single_token(self, name, expected_slug):
        result = match_speaker_to_library(name, LIBRARY_DISPLAY_NAMES)
        assert result == expected_slug


# ── Speakers NOT in library (must return None) ───────────────────────────────


class TestNoMatch:
    """
    Given  a voice library without the speaker enrolled
    When   the speaker name does not match any library entry
    Then   None is returned
    """

    @pytest.mark.parametrize(
        "name",
        [
            "Orlando Rivera",
            "Carlos Acosta",
            "Daniel Hernández",
            "Carlos Rivera",
        ],
    )
    def test_unknown_speaker_returns_none(self, name):
        result = match_speaker_to_library(name, LIBRARY_DISPLAY_NAMES)
        assert result is None


# ── Ambiguous single-token (must return None) ────────────────────────────────


class TestAmbiguousSingleToken:
    """
    Given  a voice library with two speakers sharing a first name
    When   only that first name is provided (no surname to disambiguate)
    Then   None is returned to avoid false positives
    """

    def test_ambiguous_first_name_returns_none(self):
        lib_with_two_carlos = {
            **LIBRARY_DISPLAY_NAMES,
            "carlos-acosta": "Carlos Acosta",
        }
        result = match_speaker_to_library("Carlos", lib_with_two_carlos)
        assert result is None

    def test_unique_first_name_matches_but_low_confidence(self):
        """
        Given  only one Carlos in library
        When   "Carlos" alone is provided
        Then   it matches but with low confidence (single-token match)
        """
        result = match_speaker_to_library("Carlos", LIBRARY_DISPLAY_NAMES)
        assert result == "carlos-soublette"

    def test_single_token_match_has_low_confidence(self):
        """
        Given  a single-token query that uniquely matches
        When   checked with match_with_confidence
        Then   the result has low confidence to signal the caller to verify
        """
        from speechlib.domain.name_matching import match_speaker_with_confidence

        result = match_speaker_with_confidence("Carlos", LIBRARY_DISPLAY_NAMES)
        assert result is not None
        assert result.slug == "carlos-soublette"
        assert result.confidence == "low"

    def test_full_name_match_has_high_confidence(self):
        from speechlib.domain.name_matching import match_speaker_with_confidence

        result = match_speaker_with_confidence(
            "Carlos Soublette", LIBRARY_DISPLAY_NAMES
        )
        assert result is not None
        assert result.slug == "carlos-soublette"
        assert result.confidence == "high"


# ── Surname mismatch prevents false positive ─────────────────────────────────


class TestSurnameMismatch:
    """
    Given  "Carlos Soublette" enrolled in the library
    When   "Carlos Acosta" is queried (same first name, different surname)
    Then   None is returned — surname mismatch blocks the match

    This is the critical safety property: shared first names must not
    cause cross-matching between different people.
    """

    def test_carlos_acosta_does_not_match_carlos_soublette(self):
        result = match_speaker_to_library("Carlos Acosta", LIBRARY_DISPLAY_NAMES)
        assert result is None
