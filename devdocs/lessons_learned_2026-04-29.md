# Lecciones Aprendidas — Sesión 2026-04-29

## Qué es speechlib

CLI de transcripción con diarización y reconocimiento de speakers. Branch: `refactor/speaker-domain`. Biblioteca de fingerprints en `c:\workspace\ls-work\202602 CCS TI\analisis\voices\` (datos específicos del engagement, no código).

## Glosario

- **isolation_ms**: gap mínimo en ms entre un segmento y cualquier segmento de otro speaker en el RTTM. Cero significa solapamiento temporal.
- **cluster espurio (Type B)**: pyannote agrupa audio de múltiples personas bajo un SPEAKER_XX. Irrecuperable: incluso segmentos aislados son multi-speaker.
- **Type A**: micro-segmentos de frontera (<1s) asignados al speaker equivocado.
- **por_nombrar/**: carpeta de samples sin identificar generada por el pipeline. Ejemplo: `.20260310_093950 - Gobierno/samples/por_nombrar/SPEAKER_02/`.
- **enrolar-speaker-iniciativa**: comando que copia clips de `por_nombrar/SPEAKER_XX/` a `analisis/voices/<Nombre>/`.

## Intención

Identificar y enrolar speakers sin reconocer de CCS TI (2026-03-10 a 2026-04-28). Diagnosticar contaminación de clips multi-speaker. Implementar `min_isolation_ms` para filtrar clips en zonas de transición entre speakers.

## Flujo del pipeline de speaker recognition

```
audio fuente
    ↓ pyannote diarization
diarization.rttm  (segmentos por SPEAKER_XX)
    ↓ plan_speaker_samples (domain/sample_extraction.py)
       → filtra por min_clip_duration_ms + min_isolation_ms
       → selecciona top-N por duración
SpeakerSamplePlan (qué clips extraer)
    ↓ services/extract_samples.py (I/O)
samples/por_nombrar/SPEAKER_XX/clip_01..05.wav
    ↓ usuario escucha + identifica
    ↓ enrolar-speaker-iniciativa
analisis/voices/<Nombre>/clip_01..05.wav  ← fingerprint library
    ↓ re-reconocer-iniciativa
transcript_es.vtt  (speaker labels actualizados)
```

**Parámetro nuevo esta sesión**: `min_isolation_ms=1000` en `speechlib/core_analysis.py:109`. No preexistente.

## Qué funcionó

### 1. Implementación de isolation filter

Fórmula `_isolation_ms()`: `max(0, seg.start_ms - other.end_ms, other.start_ms - seg.end_ms)` computa el gap en ms entre segmentos de speakers distintos. Con `min_isolation_ms > 0`, los segmentos con gap insuficiente se excluyen antes de aplicar top-N.

**Resultado**: 16 tests GREEN incluyendo 4 nuevos `TestIsolationFilter` en `tests/test_domain_sample_extraction.py`. Default=0 garantiza backward compatibility.

### 2. Identificación de speakers por cross-referencia de VTTs

Técnica: `rg -i "nombre" --glob "*.vtt"` sobre `c:\workspace\ls-work\202602 CCS TI\`. Hallazgos:
- Ricardo = SPEAKER_02 en 20260310 (auto-identificación en VTT: "Yo soy Ricardo, traigo algunas ideas...").
- Rita = se auto-identifica en 20260317_103037 (excluida del corpus); sin clips enrollables.

### 3. Diagnóstico: SPEAKER_02 es cluster espurio (Type B)

Evidencia: 814 de 1198 segmentos tienen `isolation=0` (solapamiento directo con otro speaker). Re-extracción manual con filtro `isolation≥1s` produjo clips contaminados igualmente.

**Lección clave**: isolation filter es **condición necesaria pero no suficiente** para Type B. No detecta si el cluster propio agrupa múltiples personas. Único diagnóstico: escucha humana.

### 4. CCS Maricel enrollada 2026-04-29

Fuente: SPEAKER_01 en `20260310_093950 - Gobierno` (identificado por escucha). 5 clips copiados a `analisis/voices/CCS Maricel/`. Aparece solo en esa sesión. Re-reconocer pendiente.

Nota factual: `min_unidentified_clip_duration_ms=500ms` en producción (no 1500ms — ese valor fue de un análisis ad-hoc local).

## Qué no funcionó

### 1. Re-extracción de Ricardo con isolation filter

Se extrajeron los 5 segmentos más largos con `isolation≥1s` (duraciones 11.8s–8.4s). Escucha directa confirmó: múltiples voces mezcladas. Conclusión: el isolation filter elimina Type A (transiciones) pero **no resuelve Type B** (cluster con múltiples personas). Fix para Type B: descartar el cluster completo.

### 2. Búsqueda de Rita en corpus válido

Rita solo se auto-identifica en la sesión 20260317, excluida del corpus por decisión del usuario. La exclusión de una sesión clausura la posibilidad de enrolar speakers que solo aparecen en ella. Documentar speakers perdidos en `ccs_ti_status.md` al excluir una sesión.

## Hallazgos técnicos clave

1. **Dos tipos de contaminación de clips**:
   - **Type A (transición)**: micro-segmentos de frontera (<1s) asignados al speaker equivocado. Fix: `min_isolation_ms=1000` los excluye.
   - **Type B (cluster espurio)**: pyannote agrupa audio de múltiples personas bajo un SPEAKER_XX. Indicador: ≥68% de segmentos con `isolation=0`. **El isolation filter no resuelve Type B.** Fix: descartar el cluster completo.

2. **VTT auto-identificaciones** (`rg -i "<nombre>" --glob "*.vtt"`) son la herramienta más confiable para mapear SPEAKER_XX → nombre real cuando los clips están contaminados. Funciona sobre VTTs originales (Zoom) y procesados por speechlib.

3. **Top-N por duración sesga hacia overlaps**: en reuniones con habla simultánea, los segmentos más largos son "envelopes" con overlaps internos. `min_isolation_ms=1000` rompe este sesgo para Type A; para Type B el cluster entero es irrecuperable.

4. **pyannote produce RTTM solapados**: dos líneas de speakers distintos pueden cubrir el mismo rango temporal (comportamiento esperado del overlap detection, no un bug).

5. **Exclusión de corpus es irreversible sobre el recognition pipeline**: excluir una sesión pierde fingerprints de speakers que solo aparecen ahí. Documentar siempre las consecuencias en `ccs_ti_status.md`.
