"""Unit tests: speechlib/device_status.py (functional core, sin I/O)."""
from speechlib.device_status import cuda_unavailable_warning


def test_cuda_unavailable_warning_points_to_torch_backend_not_mamba():
    """
    Given  el sistema detecta que CUDA no esta disponible
    When   se construye el mensaje de advertencia para el usuario
    Then   el mensaje indica CPU mode y guia al setup sancionado (torch-backend + uv),
           sin mencionar mamba ni conda
    """
    msg = cuda_unavailable_warning().lower()

    assert "mamba" not in msg, "el mensaje no debe sugerir mamba"
    assert "conda" not in msg, "el mensaje no debe sugerir conda"
    assert "cpu" in msg, "el mensaje debe avisar que corre en CPU"
    assert "torch-backend" in msg, "el mensaje debe guiar al setup sancionado"
    assert "uv sync" in msg, "el mensaje debe indicar uv sync"
