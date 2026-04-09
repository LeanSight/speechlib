"""
Funciones puras de seleccion de clips para enrollment de speakers.

Extraido de tools/enroll_speaker.py. Toda la logica de filtrado (outlier
rejection, diverse selection) opera exclusivamente sobre embeddings como
ndarrays, devolviendo indices. Cero I/O, cero Paths.

Estilo GOOS-sin-mocks: el shell (tools/enroll_speaker.py) mapea indices
de vuelta a Paths y maneja la copia de archivos.
"""

import numpy as np

from .recognition import cosine_similarity as _cosine_similarity


def reject_outlier_indices(
    embeddings: list[np.ndarray],
    sigma: float = 2.0,
) -> list[int]:
    """Devuelve indices de embeddings que NO son outliers respecto al centroide.

    Un embedding es outlier si su distancia coseno al centroide supera
    mean + sigma * std del grupo.

    Args:
        embeddings: lista de ndarrays (cualquier shape, se aplanan).
        sigma: cuantas desviaciones estandar tolerar.

    Returns:
        Lista de indices (0-based) de los embeddings no-outlier, en orden.
    """
    if len(embeddings) <= 1:
        return list(range(len(embeddings)))

    flat = [np.asarray(e, dtype=np.float64).flatten() for e in embeddings]
    centroid = np.mean(flat, axis=0)
    distances = [1.0 - _cosine_similarity(e, centroid) for e in flat]

    std = float(np.std(distances))
    if std == 0.0:
        return list(range(len(embeddings)))

    threshold = float(np.mean(distances)) + sigma * std
    return [i for i, d in enumerate(distances) if d <= threshold]


def select_diverse_indices(
    embeddings: list[np.ndarray],
    max_count: int,
) -> list[int]:
    """Selecciona indices de embeddings maximizando diversidad (greedy farthest-first).

    Empieza por el embedding mas cercano al centroide, luego agrega
    iterativamente el que maximiza la distancia minima a los ya seleccionados.

    Args:
        embeddings: lista de ndarrays.
        max_count: cuantos seleccionar.

    Returns:
        Lista de indices seleccionados, en orden de seleccion.
    """
    if len(embeddings) <= max_count:
        return list(range(len(embeddings)))

    flat = [np.asarray(e, dtype=np.float64).flatten() for e in embeddings]
    centroid = np.mean(flat, axis=0)
    distances_to_centroid = [1.0 - _cosine_similarity(e, centroid) for e in flat]

    selected = [int(np.argmin(distances_to_centroid))]

    for _ in range(max_count - 1):
        best_idx = None
        best_min_dist = -1.0

        for i, emb in enumerate(flat):
            if i in selected:
                continue
            min_dist = min(1.0 - _cosine_similarity(emb, flat[j]) for j in selected)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = i

        if best_idx is not None:
            selected.append(best_idx)

    return selected
