"""
Compatibility shims for upstream library version mismatches.

Smell 7: aislamos los monkey-patches globales aqui para mantenerlos visibles,
documentados y centralizados en lugar de dispersos por el codigo.

Uso: import this module BEFORE the libraries que lo necesitan. Tipicamente
desde el primer import de speechlib (e.g., en core_analysis.py o en
__init__.py si se centraliza ahi).

Cada patch debe documentar:
1. POR QUE existe (incompatibilidad concreta)
2. QUE espera el caller (cual es la API que rompe)
3. CUANDO puede borrarse (issue de upstream + version objetivo)
"""

import torchaudio


def patch_torchaudio_list_audio_backends() -> None:
    """Restaura torchaudio.list_audio_backends() en versiones donde fue removido.

    Origen: pyannote.audio < 4.x usa torchaudio.list_audio_backends() durante
    la inicializacion de su pipeline. torchaudio 2.x+ removio esa funcion en
    favor de torchaudio.audio_backends_utils. SpeechBrain tambien la consume
    en speechbrain/dataio/dataio.py:check_torchaudio_backend().

    Este shim devuelve ['sox'] (un backend ficticio) — los callers solo usan
    el resultado para verificar que torchaudio esta disponible, no para
    seleccionar el backend real (que es interno de torchaudio).

    Borrar cuando: pyannote.audio y speechbrain usen el API nuevo de torchaudio
    (>= 2.x). Track upstream: https://github.com/pytorch/audio/issues/3839
    """
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["sox"]


# Aplicamos el patch al importar este modulo. Cualquier consumer de speechlib
# que importe `from speechlib import compat` (o cualquier modulo que lo
# transitivamente importe) tendra el patch activo.
patch_torchaudio_list_audio_backends()
