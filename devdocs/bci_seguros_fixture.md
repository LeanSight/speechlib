# Bci Seguros Fixture — Audio de validación con SRT como ground truth

## Audio fuente

Path: `C:\workspace\@recordings\20260225 Bci Seguros Data\`

| Version | File | Size |
|---|---|---|
| raw | `20260225_091512 - Gobierno - Voz .m4a` | 47 MB |
| normalized | `20260225_091512 - Gobierno - Voz _normalized.m4a` | 66 MB |
| enhanced | `20260225_091512 - Gobierno - Voz _enhanced.m4a` | 60 MB |

**Duración**: 48.3 min (2896 s, 48 kHz mono)
**Tipo**: meeting Bci Seguros - Gobierno - Voz, ~7 speakers, audio de sala
**Ground truth**: `20260225_091512 - Gobierno - Voz _enhanced_timestamps.srt` (validado por usuario sobre versión enhanced)

**Experimentos**: `<audio_dir>/experiments/`

## Por qué es un buen fixture

Bci Seguros complementa al Alicanto fixture:

- **Bci Seguros**: audio "fácil" — meeting de 48 min, 7 speakers distintos sin contaminación de video/ruido. **Ground truth = SRT validado** (timestamps + speaker labels). Permite medir accuracy de speaker recognition con métrica cuantitativa contra una referencia trustworthy.
- **Alicanto**: audio "hard" — 3.25h, voces masculinas similares, audio de video, ruido ambiental. **Ground truth = labels manuales del usuario sobre clips de samples**. Diferente granularidad (clip-level, no timestamp-level).

Juntos cubren los dos extremos: validar el upper bound del pipeline cuando todo es ideal (Bci Seguros) y el comportamiento bajo presión (Alicanto).

## Speakers en el SRT

Total: **138 segmentos**, **7 speakers únicos**, 45.9 min de audio cubiertas (95% del total).

| Speaker | # segs | dur (s) | % time | En library? |
|---|---|---|---|---|
| Javiera | 31 | 823 | 29.9% | ❌ |
| Jolyon | 37 | 800 | 29.0% | ✓ → `BciS - Jolyon Abello` |
| Agustin | 38 | 606 | 22.0% | ✓ → `Agustin Villena` |
| Oscar | 17 | 220 | 8.0% | ❌ |
| Manuel | 11 | 179 | 6.5% | ✓ → `Manuel Olguin` |
| E | 2 | 68 | 2.5% | ❌ (etiqueta ambigua) |
| Pamela | 2 | 60 | 2.2% | ✓ → `Pamela Falconi` |

**4 de 7 speakers** en library — el pipeline debería identificarlos.
**3 de 7 desconocidos** (Javiera, Oscar, E) — deberían quedar en `por_nombrar/`.

## Métrica v2 — accuracy on covered

La métrica v1 (accuracy weighted by SRT duration, contando uncovered como error) sub-estima la performance porque cuenta el segmentation jitter como error. La v2 separa:

1. **Coverage**: % del SRT cubierto por algún segment del pipeline
2. **Accuracy on covered**: dentro de la cobertura, % atribuido al speaker correcto

Esto refleja mejor la calidad del speaker recognition porque excluye los gaps de segmentation (boundaries shifts) que no son errores reales.

## Resultados de los experimentos

### E1 — community-1 + enhanced (BASELINE)

| | Value |
|---|---|
| Tiempo | 217 s (3.6 min) sobre 48.3 min de audio = 13× realtime |
| Speakers detectados | 7 (4 identificados + 3 SPEAKER_XX) |
| Coverage | 94.9% |
| **Accuracy on covered** | **96.27%** |
| Inferred SPEAKER_01 → Javiera | 99.7% pure |
| Inferred SPEAKER_02 → E | 100% pure |
| Inferred SPEAKER_03 → Oscar | 94.9% pure |

Todos los speakers > 94% accuracy. Pamela 100%. Manuel 98.2%.

### E6 — pyannote 3.1 + enhanced (CONTROL)

| Metric | E1 (community-1) | E6 (3.1) |
|---|---|---|
| Tiempo | 217 s | 210 s |
| Coverage | 94.9% | 95.2% |
| Accuracy | 96.27% | 96.37% |
| Speakers detectados | 7 | 7 |
| Inferred SPEAKER_XX | idéntico | idéntico |

**E1 ≈ E6 sobre este audio**: diferencia 0.10 puntos porcentuales (dentro del margen estocástico).

## Hallazgos

1. **community-1 NO empeora sobre audios fáciles** — performa equivalente a 3.1 en Bci Seguros. Esto justifica adoptarlo como default sin riesgo de regresión.

2. **Coverage como métrica útil**: separar segmentation gap (jitter de boundaries) del error real de speaker recognition. Sin esa separación, métricas como "weighted accuracy" sub-estiman al pipeline.

3. **community-1 + GPU = 13× realtime**: para 48 min de audio, el pipeline corre en 3.6 min. Sin GPU se cuelga (>2h sin terminar) — el entorno torch+CUDA es CRÍTICO.

4. **Library matching trabaja excelente cuando los samples son representativos**: Pamela tiene 100% accuracy con solo 60s de audio en el SRT — los samples del library cubren bien su voz.

5. **El SPEAKER_XX naming es coherente**: cada SPEAKER_XX corresponde a UNA persona real (purity > 94%). Sin contaminación cross-speaker como sí ocurre en Alicanto.

## Diferencias clave vs Alicanto

| Aspect | Bci Seguros | Alicanto |
|---|---|---|
| Duración | 48 min | 3.25 h |
| Speakers | 7 distintos | 6 reales + audio video + ruido |
| Embedding drift | Mínimo | Significativo (3.25h) |
| Voces similares | No | Sí (Daniel/Nicolas) |
| Crosstalk | Bajo | Alto |
| Audio externo | No | Sí (video reproducido) |
| Pyannote 3.1 puro? | 96.4% accuracy | Modos de falla A+B coexistiendo |
| Necesita re-clustering? | No | Sí (alicanto_fixture.md) |

## Cómo reproducir

```bash
# 1. Verify env
python -c "import torch; print(torch.cuda.is_available())"  # True

# 2. Run E1
python temp/run_bci_seguros_e1.py

# 3. Compute metrics
python temp/metric_v2.py

# 4. Inspect
cat "C:/workspace/@recordings/20260225 Bci Seguros Data/experiments/02_E1_metrics_v2.json"
```

## Archivos clave

- `experiments/00_setup.md` — verificación de entorno + parsing del SRT
- `experiments/00_srt_ground_truth.json` — segmentos del SRT
- `experiments/00_speaker_mapping.json` — mapping SRT → library
- `experiments/01_E1_result.json` — output E1
- `experiments/02_E1_metrics_v2.json` — métricas v2 E1
- `experiments/02_E1_results.md` — análisis E1
- `experiments/06_E6_result.json` — output E6 (control 3.1)
- `experiments/06_E6_metrics_v2.json` — métricas v2 E6
- `experiments/03_E6_results.md` — análisis E1 vs E6
