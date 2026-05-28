"""
Domain tests for fuzzy speaker name -> voice library slug matching.

The voice library uses folder slugs (e.g., "acme-elena-rios") while
the --speakers CLI flag uses human names (e.g., "Elena Rios"). The matching
function bridges these two representations.

All names and organizations are fictional. No production data.
"""

import pytest
from speechlib.domain.name_matching import (
    MatchResult,
    match_speaker_to_library,
    match_speaker_with_confidence,
)


# -- Fixtures (fictional) ----------------------------------------------------

LIBRARY = {
    # org-prefixed slug, accented display_name
    "acme-elena-rios": "Elena Ríos",
    # plain slug, accented display_name
    "francisco-muñoz": "Francisco Muñoz",
    # slug with shared first name (two "Carlos" scenario)
    "carlos-delgado": "Carlos Delgado",
    # compound first name
    "juan-pablo-herrera": "Juan Pablo Herrera",
    # short surname
    "maria-sol": "María Sol",
    # common first name, unique surname
    "pedro-lagos": "Pedro Lagos",
    # another common first name
    "ana-castro": "Ana Castro",
}


# -- Exact display_name matches (with accent normalization) -------------------


class TestExactDisplayNameMatch:
    """
    Given  a voice library with display_name metadata
    When   the speaker name matches a display_name exactly (modulo accents/case)
    Then   the corresponding slug is returned
    """

    @pytest.mark.parametrize(
        "name, expected_slug",
        [
            ("Elena Ríos", "acme-elena-rios"),
            ("Elena Rios", "acme-elena-rios"),
            ("elena rios", "acme-elena-rios"),
            ("Francisco Muñoz", "francisco-muñoz"),
            ("Francisco Munoz", "francisco-muñoz"),
            ("Carlos Delgado", "carlos-delgado"),
            ("Juan Pablo Herrera", "juan-pablo-herrera"),
            ("María Sol", "maria-sol"),
        ],
    )
    def test_matches_display_name(self, name, expected_slug):
        result = match_speaker_to_library(name, LIBRARY)
        assert result == expected_slug


# -- Nickname / prefix matches -----------------------------------------------


class TestNicknameMatch:
    """
    Given  a voice library with enrolled speakers
    When   the speaker name uses a nickname that is a prefix of a known name
    Then   the correct slug is returned
    """

    @pytest.mark.parametrize(
        "name, expected_slug",
        [
            # "Ele" is prefix of "Elena"
            ("Ele Ríos", "acme-elena-rios"),
            # "Fran" is prefix of "Francisco"
            ("Fran Muñoz", "francisco-muñoz"),
        ],
    )
    def test_nickname_prefix_matches(self, name, expected_slug):
        result = match_speaker_to_library(name, LIBRARY)
        assert result == expected_slug


# -- Single-token unique match -----------------------------------------------


class TestSingleTokenMatch:
    """
    Given  a voice library where a name token uniquely identifies one speaker
    When   only that token is provided
    Then   the unique match is returned
    """

    @pytest.mark.parametrize(
        "name, expected_slug",
        [
            ("Lagos", "pedro-lagos"),
            ("Ríos", "acme-elena-rios"),
        ],
    )
    def test_unique_single_token(self, name, expected_slug):
        result = match_speaker_to_library(name, LIBRARY)
        assert result == expected_slug


# -- Speakers NOT in library (must return None) -------------------------------


class TestNoMatch:
    """
    Given  a voice library without the speaker enrolled
    When   the speaker name does not match any library entry
    Then   None is returned
    """

    @pytest.mark.parametrize(
        "name",
        [
            "Roberto Fuentes",
            "Carlos Mendoza",
            "Diana Hernández",
            "Carlos Fuentes",
        ],
    )
    def test_unknown_speaker_returns_none(self, name):
        result = match_speaker_to_library(name, LIBRARY)
        assert result is None


# -- Ambiguous single-token (must return None) --------------------------------


class TestAmbiguousSingleToken:
    """
    Given  a voice library with two speakers sharing a first name
    When   only that first name is provided (no surname to disambiguate)
    Then   None is returned to avoid false positives
    """

    def test_ambiguous_first_name_returns_none(self):
        lib_with_two_carlos = {
            **LIBRARY,
            "carlos-mendoza": "Carlos Mendoza",
        }
        result = match_speaker_to_library("Carlos", lib_with_two_carlos)
        assert result is None

    def test_unique_first_name_matches_but_low_confidence(self):
        """
        Given  only one Carlos in library
        When   "Carlos" alone is provided
        Then   it matches but with low confidence (single-token match)
        """
        result = match_speaker_to_library("Carlos", LIBRARY)
        assert result == "carlos-delgado"

    def test_single_token_match_has_low_confidence(self):
        """
        Given  a single-token query that uniquely matches
        When   checked with match_with_confidence
        Then   the result has low confidence to signal the caller to verify
        """
        result = match_speaker_with_confidence("Carlos", LIBRARY)
        assert result is not None
        assert result.slug == "carlos-delgado"
        assert result.confidence == "low"

    def test_full_name_match_has_high_confidence(self):
        result = match_speaker_with_confidence("Carlos Delgado", LIBRARY)
        assert result is not None
        assert result.slug == "carlos-delgado"
        assert result.confidence == "high"


# -- Surname mismatch prevents false positive --------------------------------


class TestSurnameMismatch:
    """
    Given  "Carlos Delgado" enrolled in the library
    When   "Carlos Mendoza" is queried (same first name, different surname)
    Then   None is returned -- surname mismatch blocks the match

    This is the critical safety property: shared first names must not
    cause cross-matching between different people.
    """

    def test_same_first_name_different_surname_returns_none(self):
        result = match_speaker_to_library("Carlos Mendoza", LIBRARY)
        assert result is None


# -- Org-prefixed slug resolution ---------------------------------------------


class TestOrgPrefixedSlug:
    """
    Given  a voice library where a slug has an org prefix (e.g., "acme-elena-rios")
    When   the display_name matches ("Elena Rios")
    Then   the org-prefixed slug is returned correctly

    The org prefix in the slug must not prevent matching against the
    display_name which does not include the org.
    """

    def test_org_prefix_does_not_block_display_name_match(self):
        result = match_speaker_to_library("Elena Rios", LIBRARY)
        assert result == "acme-elena-rios"

    def test_org_prefix_does_not_block_nickname_match(self):
        result = match_speaker_to_library("Ele Rios", LIBRARY)
        assert result == "acme-elena-rios"
