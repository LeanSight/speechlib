# Voice classification gap: conference-audio remote speakers

## Thesis

The current voice library (all 4 voices enrolled from clean in-person audio) classifies the LOCAL participant but NOT the REMOTE participants in Teams/Meet screen-recordings. Cause: remote voices reach the recording through the conference low-bitrate codec (domain shift vs their clean enrollment), so their embeddings collapse; the local participant is captured clean (local mic), so he still matches his clean enrollment. The gap is on the CAPTURE side, not the enrollment side — enrollment is clean for everyone, but only the local speaker is also captured clean.

## Glossary (one line each)

- **Speaker recognition**: deciding which known person a voice segment belongs to.
- **Enrollment**: registering a person by storing reference audio clips under a slug (e.g. `agustin-villena`).
- **Embedding**: a fixed-length numeric vector summarizing voice identity; similarity between two embeddings is a cosine-like score in 0..1.
- **Diarization**: splitting audio into clusters of "same voice" (SPEAKER_00, SPEAKER_01, ...) WITHOUT naming them.
- **Threshold**: minimum score to accept a name match (here 0.55).
- **min_margin**: minimum gap between top-1 and top-2 candidate scores to accept (here 0.1).
- **Domain shift**: train/reference audio captured under different conditions than test audio, degrading match scores.

## Recognition diagnostics (measured; threshold=0.55, min_margin=0.1)

Candidate set per session = the `import-media --speaker` hint, mapped to enrolled slugs.

### Session A — "20260623 CEN" (Teams screen-recording, 1h12m, VTT to 01:12:12, 187 cues)

Diarization: 3 dense clusters. Segment counts: SPEAKER_00=263, SPEAKER_01=265, SPEAKER_02=189.

| Cluster | agustin-villena | pamela-falconi | Decision |
|---|---|---|---|
| SPEAKER_00 | 0.7084 | 0.1425 | agustin-villena |
| SPEAKER_01 | 0.6786 | 0.1620 | agustin-villena |
| SPEAKER_02 | 0.1765 | 0.0567 | SPEAKER_02 (unresolved) |

VTT result labels: agustin-villena x178 cues, SPEAKER_02 x9.

### Session B — "20260623 CCS - Laboratorio Digital - Gobierno" (Meet screen-recording, 18m, VTT to 17:52, 152 cues)

Diarization: 3 clusters. Segment counts: SPEAKER_00=96, SPEAKER_01=92, SPEAKER_02=174.

| Cluster | agustin-villena | juan-pablo-traverso | susan-de-mello | Decision |
|---|---|---|---|---|
| SPEAKER_00 | 0.5225 | 0.3648 | 0.1581 | SPEAKER_00 (unresolved, 0.5225 < 0.55) |
| SPEAKER_01 | 0.6405 | 0.3663 | 0.0931 | agustin-villena |
| SPEAKER_02 | 0.5887 | 0.3267 | 0.0443 | agustin-villena |

### Key pattern

Only `agustin-villena` (the local/host, clean near-mic audio) scores high (0.52-0.71). Every REMOTE participant (pamela, jpt, susan) maxes far below threshold (<=0.37) in EVERY cluster. Diarization SEPARATES voices well; only NAME ATTRIBUTION fails. Multiple clusters skew to the dominant clear voice => false positives (a remote cluster spuriously matches agustin >0.6).

## The library (4 enrolled voices, from `C:/workspace/ls-work/_shared/voices`)

Enrollment domain is HOMOGENEOUS in-person events, NOT videoconference. Agustin is NOT enrolled from conference audio; he shares the same in-person workshop domain as the others. This CONTRADICTS the naive "agustin enrolled from Teams" hypothesis.

| Slug | Duration | Clips | Source sessions | Quality | Inferred domain |
|---|---|---|---|---|---|
| agustin-villena | 127.2s | 8 | 20260511 CCS Socios, 20260404 CCS CAM | good | in-person workshop (clean-mic only, NO studio supplement) |
| pamela-falconi | 77.4s | 6 | 20260422 Bci Seguros Data (5) + SPEECHLIB clean studio (1) | good + clean | in-person + clean studio |
| juan-pablo-traverso | 463.3s | 49 | 20260511 CCS Socios (4) + CCS_TI/CCS - Juan Pablo Traverso (45) | good + clean (strengthened) | in-person workshop + dedicated clean session |
| susan-de-mello | n/a | 5 | samples (generic) | good | unclear/generic samples |

No speaker shows Teams/Meet/Zoom/WhatsApp in `source_sessions`. Agustin lacks the clean/dedicated studio supplement that Pamela and JPT have.

## The audio (problematic sources, from ffprobe)

Both sources are degraded MONO streams captured from videoconference platforms. A single mixed mono track means all remote voices arrived ALREADY processed through the platform's low-bitrate voice codec, degrading embeddings before any local processing.

| Source | Platform | Original MP4 stream | Normalized source.wav |
|---|---|---|---|
| 20260623 CEN | Teams | 1 stream (idx 1), AAC, mono, 16000 Hz, 68048 bps | PCM uncompressed, mono, 16 kHz, 256000 bps |
| 20260623 CCS - Laboratorio Digital - Gobierno | Meet | 1 stream (idx 1), AAC, mono, 16000 Hz, 68011 bps | PCM uncompressed, mono, 16 kHz, 256000 bps |

No per-speaker separation tracks exist. The 16 kHz + ~68 kbit/s AAC are standard voice-codec artifacts from Teams/Meet remote-participant mixing. Normalization to PCM preserves the degraded mono stream; it does not recover lost information.

Source paths:
- `C:/workspace/@recordings/20260623 CEN/sources/LeanSight Gestión Procesos _ Microsoft Teams 2026-06-23 15-03-49.mp4`
- `C:/workspace/@recordings/20260623 CEN/sources/.LeanSight Gestión Procesos _ Microsoft Teams 2026-06-23 15-03-49/source.wav`
- `C:/workspace/@recordings/20260623 CCS - Laboratorio Digital - Gobierno/sources/Meet_ GTED + LeanSight - Thorium 2026-06-23 18-01-38.mp4`
- `C:/workspace/@recordings/20260623 CCS - Laboratorio Digital - Gobierno/sources/.Meet_ GTED + LeanSight - Thorium 2026-06-23 18-01-38/source.wav`

## Root cause

Low-bitrate videoconference voice codec + single mixed mono track degrades remote-participant embeddings below threshold. Only the local clean-audio speaker (agustin) matches; ambiguous clusters skew to him as false positives. Diarization is fine; attribution fails. This is DOMAIN SHIFT, not a threshold bug.

The asymmetry is CAPTURE-side, not enrollment-side: every voice (agustin included) is enrolled from clean in-person audio. Agustin matches because his voice in THIS recording is also clean (local mic). The remotes are enrolled clean yet CAPTURED degraded (conference codec) — clean-enrollment vs degraded-capture is the mismatch. So "agustin is special" is wrong; "agustin is the only one captured clean" is right. Any speaker recorded as a remote on a future call will fail the same way; any speaker captured locally will match.

## Why threshold tuning fails

Remotes max <=0.37, nowhere near 0.55. Lowering the threshold does not lift remotes to their correct names; it only labels more clusters as the local speaker (more false positives). No threshold separates "correct remote" from "wrong agustin" because remote embeddings carry too little identity signal.

## The fix

Enroll remote speakers from SAME-DOMAIN audio: clips cut from conference recordings (Teams/Meet mono), not only clean in-person samples. Keep a `domain` tag per enrollment clip. Optionally maintain a separate "conference variant" voice per speaker so clean and degraded embeddings do not dilute each other.

## How to reproduce / verify

Per-session diagnostics live at: `sources/.<stem>/recognition_diagnostics.json` (the dot-prefixed cache dir next to each source MP4, same stem).

Fields:
- per cluster: `scores` = map of candidate slug -> similarity score (0..1).
- `decision` = chosen slug, or the raw `SPEAKER_NN` when unresolved.
- `threshold` = accept floor (0.55 here).
- `min_margin` = required gap between top-1 and top-2 (0.1 here).

To verify: re-run recognition with the same candidate set, confirm remote slugs never exceed ~0.37 across clusters and agustin exceeds 0.55 in his clusters, matching the tables above.
