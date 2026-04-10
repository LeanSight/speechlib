# Modo Offline para speechlib

## Dependencias de red en speechlib

| Componente | Qué descarga | Repo HF / URL | Caché local |
|---|---|---|---|
| **pyannote diarization** | Pipeline community-1 (segmentation + embedding) | `pyannote/speaker-diarization-community-1` | `$HF_HOME/hub/` |
| **pyannote embedding** | Modelo ECAPA para speaker recognition | `pyannote/embedding` | `$HF_HOME/hub/` |
| **faster-whisper** | Modelo CTranslate2 (ej. `large-v3`) | `Systran/faster-whisper-large-v3` (vía HF Hub) | `$HF_HOME/hub/` |
| **OpenAI whisper** | Modelo .pt original | `https://openaipublic.azureedge.net/...` | `~/.cache/whisper/` |
| **ClearVoice MossFormer2** | Checkpoint MossFormer2_SE_48K | `alibabasglab/MossFormer2_SE_48K` (HF) | `ClearerVoice-Studio/clearvoice/checkpoints/` |
| **AssemblyAI** | N/A — API cloud pura | `api.assemblyai.com` | No aplica (incompatible offline) |
| **HuggingFace transformers** | Modelo ASR si se usa `huggingface_model()` | Variable según modelo | `$HF_HOME/hub/` |

**Nota**: El path default de speechlib usa `model_cache.py` que setea `HF_HOME` y `HF_HUB_CACHE` a `~/.cache/huggingface` (o `$XDG_CACHE_HOME/huggingface`).

## Variables de entorno para modo offline

```bash
# Bloquea TODAS las llamadas HTTP del HuggingFace Hub
export HF_HUB_OFFLINE=1

# Bloquea descargas de la librería transformers (huggingface_model path)
export TRANSFORMERS_OFFLINE=1

# Token HF — necesario solo para la descarga inicial (modelos gated)
export HF_TOKEN=hf_...

# Cache centralizado (speechlib ya lo setea en model_cache.py)
export HF_HOME=~/.cache/huggingface
export HF_HUB_CACHE=~/.cache/huggingface/hub
```

Con `HF_HUB_OFFLINE=1`, cualquier intento de red lanza `OfflineModeIsEnabled` en vez de quedarse colgado esperando timeout.

## Modelos a pre-descargar

Ejecutar **con internet** antes de ir offline:

```python
# 1. Pipeline diarización (~200 MB total: segmentation + embedding internos)
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1", token="hf_..."
)

# 2. Embedding model para speaker recognition (~27 MB)
from pyannote.audio import Model
model = Model.from_pretrained("pyannote/embedding", use_auth_token="hf_...")

# 3. faster-whisper (large-v3 ≈ 3.1 GB, base ≈ 150 MB)
from faster_whisper import WhisperModel
WhisperModel("large-v3", device="cpu", compute_type="float32")

# 4. MossFormer2 — descarga automática al primer uso
#    Verificar que exista: ClearerVoice-Studio/clearvoice/checkpoints/MossFormer2_SE_48K/
#    Si no, descargar manualmente de: https://huggingface.co/alibabasglab/MossFormer2_SE_48K
from clearvoice import ClearVoice
cv = ClearVoice(task='speech_enhancement', model_names=['MossFormer2_SE_48K'])
```

| Modelo | Tamaño aprox. |
|---|---|
| `pyannote/speaker-diarization-community-1` | ~200 MB |
| `pyannote/embedding` | ~27 MB |
| `Systran/faster-whisper-large-v3` | ~3.1 GB |
| `Systran/faster-whisper-base` | ~150 MB |
| `alibabasglab/MossFormer2_SE_48K` | ~160 MB |

## Cambios de código necesarios

### 1. `diarization.py` — ya funciona offline si el cache está poblado
`Pipeline.from_pretrained` pasa por HF Hub; con `HF_HUB_OFFLINE=1` usa el cache local. **No necesita cambios** si los modelos ya están descargados.

### 2. `speaker_recognition.py` — ídem
`Model.from_pretrained("pyannote/embedding")` usa el mismo mecanismo HF Hub. Sin cambios.

### 3. `transcribe.py` — WhisperModel acepta path local
`WhisperModel(model_size)` intenta descargar si no está en cache. Para forzar offline, dos opciones:
- **Opción A** (sin cambios de código): setear `HF_HUB_OFFLINE=1` — falla rápido si no está en cache.
- **Opción B** (más robusto): pasar path local explícito en vez de string `"large-v3"`.

### 4. `enhance_audio.py` — ClearVoice descarga por su cuenta
ClearVoice no usa HF Hub estándar; descarga checkpoints a `clearvoice/checkpoints/`. **Requiere pre-descarga manual** del checkpoint antes de ir offline.

### 5. AssemblyAI — incompatible con offline
`assembly_ai_model()` es una API cloud. Documentar que este path no funciona offline.

### 6. Cambio sugerido en `__main__.py`
Agregar flag `--offline` que setee las variables de entorno antes de importar cualquier librería ML:

```python
if args.offline:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
```

## Checklist pre-vuelo

- [ ] Máquina con internet: ejecutar el script de pre-descarga de la sección anterior
- [ ] Verificar cache: `python -c "from speechlib.model_cache import print_cache_info; print_cache_info()"`
- [ ] Verificar checkpoint ClearVoice: confirmar que existe `clearvoice/checkpoints/MossFormer2_SE_48K/`
- [ ] Setear variables: `HF_HUB_OFFLINE=1` y `TRANSFORMERS_OFFLINE=1`
- [ ] Test de humo: correr speechlib con un audio corto y verificar que no hay errores de conexión
- [ ] No usar `assembly_ai_model()` — requiere internet obligatoriamente
- [ ] Opcional: agregar `--offline` flag al CLI (ver cambio sugerido arriba)
