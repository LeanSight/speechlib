# IMPROVE — Runtime warnings de speechlib

> Audiencia: mantenedor de speechlib.
> Origen: estos cuatro warnings salieron a la luz mientras se perfilaba el GPU
> worker de `biz_long_term_memory`. Referencia cruzada del perfilado:
> `C:\workspace\dev\biz_long_term_memory\devdocs\experiments\speechlib-warm\EXPERIMENT-speechlib-warm-vs-cold.md`.

## Contexto

Durante el perfilado warm-vs-cold del GPU worker aparecieron cuatro warnings de
runtime al ejecutar el análisis de speechlib. Ninguno detiene el pipeline, pero
tres son ruido cosmético y uno tiene un ángulo de correctitud. Este documento
detalla, para cada uno: el mensaje, dónde se dispara (archivo:línea de la
librería + versión, y el call site de speechlib), la causa raíz, la severidad,
el fix concreto recomendado, y el caveat de GPU para TF32.

Entorno observado: `torch 2.12.0+cu126`, `torchaudio 2.11.0+cu126`,
`pyannote_audio 4.0.4`. GPU de desarrollo: **RTX 2070 Super Max-Q = Turing
(sm_75)** — sin TF32 (TF32 es Ampere sm_80+). Triton NO instalado (no hay wheel
oficial de Windows).

Punto de inserción común para todos los fixes: en `core_analysis.py`, justo
después de `from . import compat` (la única puerta de import antes de cargar los
modelos). Ahí se concentran los `warnings.filterwarnings(...)`, los ajustes de
`logging`, y cualquier política de TF32.

---

## Warning 1 — triton not found (flop counter)

**Mensaje:**

```
triton not found; flop counting will not work for triton kernels
```

**Dónde se dispara:**
`torch/utils/flop_counter.py:29` (torch 2.12.0+cu126). Se emite vía
`log.warning(...)` dentro del bloque a nivel de módulo
`try: from triton.runtime.jit import JITFunction except ImportError:`
(`flop_counter.py:24-30`), condicionado a que `torch.version.cuda` esté seteado.

**Call site de speechlib:** ninguno directo. `torch.utils.flop_counter` se
importa transitivamente (internals de pyannote/torch lo tocan). speechlib nunca
llama al FLOP counter; el warning es puro efecto colateral de un build de torch
con CUDA habilitado pero sin triton en Windows.

**Causa raíz:** al importar `torch.utils.flop_counter`, este intenta
`from triton.runtime.jit import JITFunction`. Triton NO está instalado en este
venv (no hay `triton-*.dist-info`). Como `torch.version.cuda` está seteado
(build cu126), la rama `ImportError` loguea el warning. Triton no tiene wheel
oficial de Windows, así que su ausencia es legítima.

**Severidad:** benigno-cosmético.

**Fix recomendado:** dejar triton ausente TAL CUAL (no intentar instalar triton
en Windows — no hay wheel oficial). Suprimir el ruido subiendo el nivel de log
de ese logger específico, en `core_analysis.py` justo después de
`from . import compat`:

```python
import logging
logging.getLogger('torch.utils.flop_counter').setLevel(logging.ERROR)
```

IMPORTANTE: esto es un `logging.Logger.warning`, NO un `warnings.warn`. Un
filtro `warnings.filterwarnings` NO lo silencia — hay que silenciarlo vía la API
de `logging`. No bajar el nivel por debajo de `ERROR` globalmente; limitarlo
solo a ese nombre de logger.

**Caveat GPU (TF32):** independiente de la arquitectura. El FLOP counting vía
triton es una feature de profiling/debug que speechlib no usa en ninguna GPU; el
warning es idéntico en Turing (RTX 2070 Super Max-Q) y en Ampere+. Sin impacto en
exactitud ni performance en ninguna arquitectura.

---

## Warning 2 — bits_per_sample no soportado por TorchCodec

**Mensaje:**

```
The 'bits_per_sample' parameter is not directly supported by TorchCodec AudioEncoder.
```

**Dónde se dispara:**
`torchaudio/__init__.py:178` -> `torchaudio/_torchcodec.py` en
`save_with_torchcodec()`, `warnings.warn` en `_torchcodec.py:270-277`
(torchaudio 2.11.0+cu126). `torchaudio.save()` ahora reenvía a
`save_with_torchcodec()` (`__init__.py:178`), que avisa cada vez que
`bits_per_sample` no es `None`.

**Call sites de speechlib:** `torchaudio.save(..., bits_per_sample=16)` en:
- `speechlib/resample_to_16k.py:17`
- `speechlib/loudnorm.py:36`
- `speechlib/audio_utils.py:12`
- `speechlib/audio_utils.py:39`
- `speechlib/re_encode.py:21`

(`convert_to_mono.py:17` y `convert_to_wav.py:15` llaman a `save` SIN
`bits_per_sample`, por eso no avisan.)

**Causa raíz:** torchaudio 2.10+ enruta `torchaudio.save` a través del
`AudioEncoder` de TorchCodec, que ignora `bits_per_sample` (la profundidad de
bits de salida se infiere de la extensión/codec del archivo). Pasar
`bits_per_sample=16` emite un `UserWarning` por cada llamada. speechlib TIENE un
shim de compat (`speechlib/compat.py` `patch_torchaudio_torchcodec`) que
reemplaza `torchaudio.save` por una implementación basada en `wave` de la stdlib
que respeta `bits_per_sample` — pero ese shim SONDEA torchcodec en
`compat.py:50-55` (`from torchcodec.decoders import AudioDecoder`) y retorna
temprano en L53 si torchcodec importa limpio. En esta máquina torchcodec
funciona, así que el parche se omite y corre el `save` real respaldado por
torchcodec, emitiendo el warning dos veces por archivo (una durante el
pre-procesado resample/loudnorm, otra al escribir la salida). El warning es
benigno: el WAV PCM de 16 bits se produce igual correctamente porque la
extensión `.wav` ya implica PCM de 16 bits.

**Severidad:** benigno-cosmético.

**Fix recomendado:** dos fixes válidos; preferir el primero.

(1) Dejar de pasar el kwarg no-op: quitar `bits_per_sample=16` de las 5
llamadas a `torchaudio.save` (`resample_to_16k.py:17`, `loudnorm.py:36`,
`audio_utils.py:12` y `:39`, `re_encode.py:21`). PCM de 16 bits es el default
para un target `.wav` bajo TorchCodec, así que la salida no cambia y el warning
desaparece en la fuente. Nota: `_save_with_wave` de `compat.py` ya tiene default
`bits_per_sample=16`, así que el camino de fallback sigue correcto.

(2) Si hay que conservar el kwarg para el camino de fallback de compat, suprimir
solo este mensaje al arranque (`core_analysis.py` tras el import de compat):

```python
warnings.filterwarnings(
    'ignore',
    message="The 'bits_per_sample' parameter is not directly supported",
    category=UserWarning,
)
```

NO silenciar `UserWarning` de forma amplia — limitar por `message`.

**Caveat GPU (TF32):** independiente de la arquitectura. `torchaudio.save` corre
sobre tensores de CPU y usa FFmpeg bajo TorchCodec; comportamiento idéntico en
Turing y Ampere+. Sin involucramiento de GPU.

---

## Warning 3 — TF32 deshabilitado (pyannote reproducibility)

**Mensaje:**

```
TensorFloat-32 (TF32) has been disabled as it might lead to reproducibility
issues and lower accuracy. It can be re-enabled by calling
torch.backends.cuda.matmul.allow_tf32 = True /
torch.backends.cudnn.allow_tf32 = True.
See https://github.com/pyannote/pyannote-audio/issues/1370
```

**Dónde se dispara:**
`pyannote/audio/utils/reproducibility.py:74` (`ReproducibilityWarning`), dentro
de `fix_reproducibility(device)` en `reproducibility.py:68-83`
(pyannote_audio 4.0.4). Solo se dispara cuando `device.type == 'cuda'` Y TF32
estaba habilitado en ese momento; entonces setea
`torch.backends.cuda.matmul.allow_tf32 = False` y
`torch.backends.cudnn.allow_tf32 = False` y avisa.

**Call site de speechlib:** `speechlib/diarization.py:24`
`pipeline.to(torch.device('cuda'))` dentro de `get_diarization_pipeline()`
(`lru_cache`, llamado una vez desde `core_analysis`). Mover el pipeline de
pyannote a CUDA invoca `fix_reproducibility`, que deshabilita TF32 globalmente
para todo el proceso.

**Causa raíz:** pyannote 4.x deshabilita deliberadamente TF32 en CUDA para
mantener la diarización determinista/exacta, y lo anuncia una vez. El efecto
colateral es GLOBAL: después de la diarización, TF32 queda apagado para
cualquier matmul/conv CUDA posterior en el mismo proceso (ej. faster-whisper si
usara un camino CUDA de torch). Es un warning informativo de una sola vez, no un
error. speechlib no setea ninguna política de TF32 propia (grep de
`allow_tf32`/`set_float32_matmul`/`TF32` en speechlib = sin matches), así que
gana el default de pyannote.

**Severidad:** reproducibilidad.

**Fix recomendado:** hacer explícita la intención de speechlib en vez de
dejarla a un efecto colateral de la librería. Decidir una política y setearla una
vez al arranque en `core_analysis.py` (tras el import de compat, antes de que
corra `get_diarization_pipeline`). Para transcripts de memoria institucional, la
exactitud/reproducibilidad pesa más que un pequeño porcentaje de velocidad de
matmul, así que ACEPTAR el disable y solo silenciar el anuncio:

```python
from pyannote.audio.utils.reproducibility import ReproducibilityWarning
warnings.filterwarnings('ignore', category=ReproducibilityWarning)
```

Si en cambio se quiere TF32 ON para el trabajo de whisper/otro CUDA que corre
DESPUÉS de la diarización en una caja Ampere+, re-habilitarlo explícitamente
post-diarización:

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

Hacerlo conscientemente y documentarlo; no depender de la mutación global
implícita de pyannote. **Default recomendado:** mantener TF32 deshabilitado
(coincide con la intención de pyannote) y solo suprimir el warning.

**Caveat GPU (TF32):** esta máquina es **RTX 2070 Super Max-Q = Turing
(sm_75)**, que NO tiene tensor cores TF32 — TF32 es feature de Ampere (sm_80).
Por lo tanto tanto deshabilitar COMO re-habilitar TF32 acá son NO-OPs: cero
efecto de perf o exactitud en Turing, el warning es puramente cosmético en esta
caja. SÍ IMPORTA en Ampere+ (RTX 30xx/40xx, A100, etc.): ahí, que pyannote
deshabilite TF32 ralentiza (y cambia la numérica de) cualquier matmul/conv CUDA
posterior en el mismo proceso. Por eso, cualquier código de re-habilitación de
TF32 agregado por performance debe estar guardado a Ampere+ (ej. chequear
`torch.cuda.get_device_capability()[0] >= 8`) y es irrelevante en la máquina de
desarrollo Turing actual.

---

## Warning 4 — std(): degrees of freedom <= 0 (StatsPool)

**Mensaje:**

```
std(): degrees of freedom is <= 0. Correction should be strictly less than the
reduction factor (input numel divided by output numel).
(Triggered internally at .../ATen/native/ReduceOps.cpp:1879.)
```

**Dónde se dispara:**
`pyannote/audio/models/blocks/pooling.py:103` (pyannote_audio 4.0.4):
`std = sequences.std(dim=-1, correction=1)` dentro de `StatsPool.forward` (la
rama `weights is None` en `pooling.py:101-104`). El warning lo levanta ATen
`std()` cuando la dimensión reducida tiene largo 1 y la corrección de Bessel
`correction=1` hace `dof = N-1 = 0`.

**Call site de speechlib:** el forward pass de diarización.
`get_diarization_pipeline(...).apply()` (pipeline construido en
`speechlib/diarization.py:20-22`, corrido desde `core_analysis`) alimenta
segmentos de speaker cortos/casi-silenciosos al modelo de speaker-embedding,
cuyo `StatsPool` computa un `std` insesgado sobre un único frame restante.

**Causa raíz:** cuando un segmento diarizado es tan corto (o tan reducido por el
subsampling temporal/VAD del modelo de embedding) que solo UN frame llega a
`StatsPool`, `std(dim=-1, correction=1)` tiene `dof = N-1 = 0`, así que ATen
avisa y devuelve `NaN` para la mitad `std` del estadístico `(mean, std)` de ese
segmento. Es inherente a alimentar segmentos de 1 frame al `StatsPool` de
pyannote; en audio real de reuniones siempre hay turnos breves/superpuestos/
silenciosos, así que se dispara una vez por corrida. Es interno a pyannote —
speechlib no puede cambiar la llamada a `std()` sin parchear la librería.

**Severidad:** correctitud.

**Fix recomendado:** speechlib no puede editar `pooling.py`, y la matemática
subyacente es de pyannote, así que la acción práctica es doble.

(a) Suprimir el ruido al arranque (`core_analysis.py` tras el import de compat):

```python
warnings.filterwarnings(
    'ignore',
    message='std\\(\\): degrees of freedom is <= 0',
    category=UserWarning,
)
```

(b) Reducir la frecuencia con que se dispara guardando la diarización contra
inputs ultra-cortos: el `StatsPool` de pyannote produce `std` NaN para segmentos
de 1 frame, así que antes de correr la diarización, saltar/redondear-padear
clips por debajo de una duración mínima (el spike de 5s ya muestra al VAD
removiendo el clip entero: `VAD filter removed 00:05.000 of audio`).
Concretamente, en el camino de pre-procesado que entrega audio a
`get_diarization_pipeline`, cortocircuitar segmentos bajo un piso pequeño (ej.
`< ~0.5s` de audio con voz) en vez de diarizarlos.

Tratar esto como relevante a correctitud (embeddings NaN pueden perturbar el
clustering) — no solo ocultarlo sin el guard de segmento corto. No cambiar
`correction` globalmente.

**Caveat GPU (TF32):** independiente de la arquitectura. El problema de
`std()`/dof es una condición de forma de tensor/numérica en `StatsPool`,
idéntica en CPU, Turing y Ampere+. El mensaje de `ReduceOps.cpp` viene del kernel
ATen de CPU/CUDA sin importar la arquitectura; TF32 no tiene relación. Sin
comportamiento específico de Turing.

---

## Priorización

| # | Warning | Severidad | Ángulo | Acción |
|---|---------|-----------|--------|--------|
| 1 | triton not found | benigno-cosmético | ninguno | suprimir vía `logging` (no `warnings`) |
| 2 | bits_per_sample no soportado | benigno-cosmético | ninguno | quitar kwarg no-op (preferido) o `filterwarnings` por mensaje |
| 3 | TF32 deshabilitado | reproducibilidad | perf en Ampere+, NO-OP en Turing | hacer política explícita; default = mantener off + suprimir |
| 4 | std(): dof <= 0 | correctitud | embeddings NaN pueden perturbar clustering | suprimir + guard de segmento corto |

**Puros cosmético-suprimir (sin ángulo de perf/correctitud):**
- **#1 (triton):** suprimir es la solución completa, no hay nada más que hacer.
- **#2 (bits_per_sample):** la salida ya es correcta; quitar el kwarg es el fix
  limpio en la fuente.

**Con ángulo de perf/correctitud (no solo ocultar):**
- **#3 (TF32):** ángulo de PERFORMANCE, pero solo en Ampere+. En la Turing
  actual es NO-OP, puro cosmético. Hacer la política explícita para no depender
  de la mutación global de pyannote; cualquier re-habilitación debe ir guardada a
  `get_device_capability()[0] >= 8`.
- **#4 (std dof):** ángulo de CORRECTITUD. Suprimir sin el guard de segmento
  corto deja embeddings NaN entrando al clustering. El guard de duración mínima
  (`< ~0.5s` de audio con voz) es la parte que importa; la supresión es solo
  cosmética encima de eso.

**Orden sugerido de implementación:** #4 (correctitud, con guard) > #3 (política
explícita de TF32, relevante al portar a Ampere+) > #2 (limpieza en la fuente) >
#1 (supresión cosmética).
