"""
Smell 6: average_embeddings es una funcion pura que vive en el dominio.

Antes: la logica de promediar embeddings (con filtro de NaN) estaba inline
en _compute_averaged_embeddings_per_tag de core_analysis.py:229. El comentario
admitia "mirroring de la logica canonica de speaker_recognition" — concepto
duplicado.

Despues: una sola funcion pura en speechlib/domain/recognition.py que
ambos sites pueden reusar.

Tests puros: solo numpy arrays, sin mocks, sin I/O.
"""

import numpy as np
import pytest


def test_average_embeddings_returns_mean_of_clean_embeddings():
    from speechlib.domain.recognition import average_embeddings

    embs = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    result = average_embeddings(embs)

    expected = np.array([1/3, 1/3, 1/3])
    assert np.allclose(result, expected)


def test_average_embeddings_filters_nan_embeddings():
    """Embeddings con NaN deben ser descartados antes del promedio.
    Reproduce el filtro NaN del legacy."""
    from speechlib.domain.recognition import average_embeddings

    embs = [
        np.array([1.0, 0.0, 0.0]),
        np.array([float("nan"), 0.0, 0.0]),  # filtrado
        np.array([0.0, 1.0, 0.0]),
    ]
    result = average_embeddings(embs)

    # Promedio de los 2 validos
    expected = np.array([0.5, 0.5, 0.0])
    assert np.allclose(result, expected)


def test_average_embeddings_returns_none_when_all_nan():
    """Si todos los embeddings tienen NaN, no hay nada que promediar."""
    from speechlib.domain.recognition import average_embeddings

    embs = [
        np.array([float("nan"), 0.0]),
        np.array([float("nan"), 1.0]),
    ]
    result = average_embeddings(embs)
    assert result is None


def test_average_embeddings_returns_none_when_empty():
    from speechlib.domain.recognition import average_embeddings

    assert average_embeddings([]) is None


def test_average_embeddings_handles_single_embedding():
    from speechlib.domain.recognition import average_embeddings

    emb = np.array([0.5, 0.5, 0.5])
    result = average_embeddings([emb])
    assert np.allclose(result, emb)


def test_average_embeddings_does_not_mutate_input():
    """Funcion pura: la lista de input no se modifica."""
    from speechlib.domain.recognition import average_embeddings

    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])
    embs = [e1, e2]

    average_embeddings(embs)

    assert len(embs) == 2
    assert np.array_equal(embs[0], np.array([1.0, 0.0]))
    assert np.array_equal(embs[1], np.array([0.0, 1.0]))


def test_average_embeddings_flattens_multidimensional_embeddings():
    """Acepta embeddings con shape (1, N) ademas de (N,)."""
    from speechlib.domain.recognition import average_embeddings

    embs = [
        np.array([[1.0, 0.0, 0.0]]),  # shape (1, 3)
        np.array([0.0, 1.0, 0.0]),     # shape (3,)
    ]
    result = average_embeddings(embs)
    expected = np.array([0.5, 0.5, 0.0])
    assert np.allclose(result, expected)
