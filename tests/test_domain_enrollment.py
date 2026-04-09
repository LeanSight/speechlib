"""
Unit tests: funciones puras de enrollment (domain/enrollment.py).

Estilo GOOS-sin-mocks: solo numpy, verificacion de salida. Cero I/O.
"""

import numpy as np
import pytest

from speechlib.domain.enrollment import reject_outlier_indices, select_diverse_indices


def _unit(*components: float) -> np.ndarray:
    v = np.array(components, dtype=np.float64)
    return v / np.linalg.norm(v)


class TestRejectOutlierIndices:
    def test_single_embedding_always_kept(self):
        assert reject_outlier_indices([_unit(1, 0, 0)]) == [0]

    def test_empty_returns_empty(self):
        assert reject_outlier_indices([]) == []

    def test_identical_embeddings_all_kept(self):
        embs = [_unit(1, 0, 0)] * 5
        assert reject_outlier_indices(embs) == [0, 1, 2, 3, 4]

    def test_one_outlier_rejected(self):
        """4 embeddings apuntando similar + 1 opuesto: el opuesto es outlier."""
        cluster = [_unit(1, 0.1 * i, 0) for i in range(4)]
        outlier = _unit(-1, 0, 0)
        embs = cluster + [outlier]
        indices = reject_outlier_indices(embs, sigma=1.5)
        assert 4 not in indices  # outlier rechazado
        assert len(indices) >= 3  # cluster mayormente preservado

    def test_returns_sorted_indices(self):
        embs = [_unit(1, 0, 0), _unit(0.9, 0.1, 0), _unit(0.8, 0.2, 0)]
        indices = reject_outlier_indices(embs)
        assert indices == sorted(indices)


class TestSelectDiverseIndices:
    def test_fewer_than_max_returns_all(self):
        embs = [_unit(1, 0, 0), _unit(0, 1, 0)]
        assert select_diverse_indices(embs, max_count=5) == [0, 1]

    def test_exact_max_returns_all(self):
        embs = [_unit(1, 0, 0), _unit(0, 1, 0), _unit(0, 0, 1)]
        assert select_diverse_indices(embs, max_count=3) == [0, 1, 2]

    def test_selects_diverse_not_duplicate(self):
        """3 clusters de 2 cada uno, max_count=3: debe seleccionar 1 de cada cluster."""
        a1, a2 = _unit(1, 0, 0), _unit(0.99, 0.01, 0)
        b1, b2 = _unit(0, 1, 0), _unit(0.01, 0.99, 0)
        c1, c2 = _unit(0, 0, 1), _unit(0.01, 0, 0.99)
        embs = [a1, a2, b1, b2, c1, c2]
        indices = select_diverse_indices(embs, max_count=3)
        assert len(indices) == 3
        # No debe seleccionar los 2 de un mismo cluster
        assert not ({0, 1} <= set(indices))
        assert not ({2, 3} <= set(indices))
        assert not ({4, 5} <= set(indices))

    def test_first_selected_is_closest_to_centroid(self):
        embs = [_unit(0, 0, 1), _unit(1, 0, 0), _unit(0.5, 0.5, 0)]
        # Centroid is roughly (0.5, 0.17, 0.33) — closest is embs[2]
        indices = select_diverse_indices(embs, max_count=2)
        assert indices[0] == 2
