"""Mensajes puros sobre el estado del device de torch para el usuario.

Functional core: sin I/O, sin importar torch. La decision de imprimir y la
consulta a torch.cuda viven en la shell (core_analysis); aqui solo se construye
el texto, asi es testeable sin GPU ni mocks.
"""


def cuda_unavailable_warning() -> str:
    """Advertencia cuando CUDA no esta disponible (CPU mode)."""
    return (
        "WARN: CUDA no disponible — CPU mode (lento). "
        "Ejecuta `torch-backend` en la raiz del repo y luego `uv sync` para wheels GPU."
    )
