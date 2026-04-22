from pathlib import Path
from pydantic import BaseModel, ConfigDict

_LIMPIO_SUFFIX = "_limpio"


class AudioState(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_path: Path
    working_path: Path
    is_wav: bool = False
    is_mono: bool = False
    is_16bit: bool = False
    is_16khz: bool = False
    is_normalized: bool = False
    is_enhanced: bool = False

    @property
    def artifacts_dir(self) -> Path:
        """Carpeta oculta junto al source para todos los artefactos del pipeline.

        Ejemplo: /rec/Voz 260320.m4a → /rec/.Voz 260320/

        Fallback: si el source es `<stem>_limpio.<ext>` (output del pipeline
        publicado al lado del source) y el cache `.<stem>_limpio/` no existe
        pero `.<stem>/` sí, resuelve al cache sin sufijo. Permite re-invocar
        subcomandos después de borrar el original para ahorrar espacio.
        """
        parent = self.source_path.parent
        stem = self.source_path.stem.strip()
        direct = parent / f".{stem}"
        if direct.exists():
            return direct
        if stem.endswith(_LIMPIO_SUFFIX):
            fallback = parent / f".{stem[: -len(_LIMPIO_SUFFIX)]}"
            if fallback.exists():
                return fallback
        return direct
