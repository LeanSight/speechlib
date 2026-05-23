# CLAUDE.md — speechlib

Speech processing library: diarization (pyannote), transcription (faster-whisper), and speaker recognition. Outputs VTT transcripts with real speaker names.

## Setup

```bash
uv sync                              # creates .venv with torch+CUDA
uv run pytest tests/ -k "not e2e"    # verify
```

- `HF_TOKEN` env var required (HuggingFace access token for pyannote gated models). Override per-command: `--token <value>`.
- FFmpeg must be in PATH.
- GPU optional — CPU fallback is automatic (~10x slower).

## Running

```bash
# 1. Full pipeline: preprocess → diarize → transcribe → recognize
uv run python -m speechlib run audio.m4a --voices-folder voices/ --speakers "Ana,Bruno" -v

# 2. Review .<stem>/speaker_map_suggestions.json, then create .<stem>/speaker_map.json

# 3. Apply confirmed mapping → produces <stem>_limpio.vtt next to source
uv run python -m speechlib confirm audio.m4a

# Re-run recognition only (fast):  recognize audio.m4a --voices-folder voices/ --force
# Score matrix (read-only):        diagnose audio.m4a --voices-folder voices/
```

### Key flags (`run`)

| Flag | Default | Purpose |
|------|---------|---------|
| `--voices-folder` | none | Folder with speaker voice samples |
| `--speakers` | none | Comma-separated expected attendees (filters voice library) |
| `--language` | `es` | Language code |
| `--model` | `large-v3-turbo` | Whisper model size |
| `--output-format` | `vtt` | `vtt` or `txt` |
| `--skip-enhance` | false | Skip ClearVoice speech enhancement |
| `--compress` | false | Generate compressed `_limpio.m4a` alongside VTT |
| `--initial-prompt` | none | Context text biasing Whisper (domain terms, jargon) |
| `--hotwords` | none | Logit bias terms: CSV or `@path/to/file` (one per line) |
| `-v` | false | Verbose/debug logging |

### Artifacts

All intermediate files live in `.<stem>/` next to source (e.g., `audio.m4a` → `.audio/`):

- `16k.wav` — preprocessed audio
- `diarization.rttm` — pyannote output
- `transcript_<lang>.vtt` — internal VTT
- `speaker_map_suggestions.json` — top-3 candidates per speaker with scores and `recommended`
- `speaker_map.json` — **user-created** confirmed mapping
- `recognition_diagnostics.json` — full score matrix
- `samples/<Name>/` or `samples/por_nombrar/SPEAKER_XX/` — extracted speaker clips

Final output: `<stem>_limpio.vtt` published next to source.

### speaker_map.json format

```json
{"SPEAKER_00": "Ana", "SPEAKER_01": "Bruno"}
```

Keys = diarization tags, values = real names. Unmapped tags stay as `[SPEAKER_XX]` in output.

### Voice folder structure

```
voices/
├── Ana/
│   ├── sample_1.wav    # 5–30s per clip recommended
│   └── sample_2.wav
└── Bruno/
    └── recording.wav
```

Subdirectories starting with `_` are skipped. More samples per speaker = more robust embeddings.

## Testing

```bash
uv run pytest tests/                    # all tests
uv run pytest tests/ -k "not e2e"       # skip GPU/network tests
uv run pytest -m e2e                    # e2e only (needs HF_TOKEN + audio fixtures)
```

## Architecture

```
speechlib/
├── domain/          # pure value objects (no I/O)
├── services/        # application services (I/O orchestration)
├── tools/           # CLI utilities (batch_process, diagnose, enroll)
├── core_analysis.py # pipeline orchestration
├── audio_state.py   # AudioState: frozen Pydantic model tracking preprocessing flags
├── enhance_audio.py # ClearVoice speech enhancement
└── __main__.py      # Typer CLI
```

Pipeline: source → WAV → mono → 16-bit → 16kHz → loudnorm → diarize → transcribe → recognize → suggest → confirm → publish VTT.

`AudioState` is frozen (fields: `source_path`, `working_path`, `is_wav`, `is_mono`, `is_16bit`, `is_16khz`, `is_normalized`, `is_enhanced`). Updates via `model_copy(update={...})`. Source audio is never modified.

## Where to make changes

- **New preprocessing step**: add a module in `speechlib/` (like `resample_to_16k.py`), add a bool flag to `AudioState` in `audio_state.py`, wire it in `core_analysis.py` pipeline sequence. Write acceptance test first.
- **Speaker recognition logic**: `speechlib/speaker_recognition.py` (scoring, thresholds), `speechlib/domain/recognition.py` (value objects), `speechlib/services/` (orchestration).
- **CLI flags**: `speechlib/__main__.py` — Typer decorators define flags and defaults.
- **Transcription**: `speechlib/core_analysis.py` calls `faster_whisper` via `_transcribe_segments()`.

## Conventions

- **One-piece-flow**: one behavior slice per commit (RED → GREEN → REFACTOR → commit).
- **Immutable state**: `AudioState` is frozen. Preprocessing returns new copies.
- **Domain purity**: `speechlib/domain/` has zero I/O. No mocks on domain in tests.
- **User-controlled naming**: pipeline suggests names via `speaker_map_suggestions.json` but never auto-applies. User creates `speaker_map.json`, then runs `confirm`.

## CUDA / torch

PyTorch CUDA wheels come from a dedicated index in `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu126", marker = "sys_platform == 'linux' or sys_platform == 'win32'" }]
```

No conda/mamba needed. macOS gets CPU-only wheels from PyPI.

## Gotchas

- `faster_whisper.transcribe(hotwords=...)` expects `str`, not `list[str]`
- AssemblyAI `keyterms_prompt` rejected for non-English — use `word_boost` (marginal for es-CL)
- `diarization.rttm` cache NOT auto-invalidated when `--speakers` count changes — delete `.<stem>/diarization.rttm` manually
- `clearvoice` imported via `sys.path.insert` in `enhance_audio.py` from `c:\workspace\#dev\ClearerVoice-Studio\clearvoice`. Its dep `yamlargparse` is declared in speechlib's pyproject.toml.
- **ClearerVoice-Studio clone on Windows**: the repo contains `train/target_speaker_extraction/data/wsj0_2mix/train/aux.scp` — `aux` is a reserved device name on NTFS (like `con`, `prn`, `nul`), so `git clone` fails. Fix: use sparse checkout to pull only the `clearvoice/` subfolder (the only part speechlib needs):
  ```bash
  git clone --sparse --filter=blob:none <repo-url> ClearerVoice-Studio
  cd ClearerVoice-Studio
  git sparse-checkout set clearvoice
  ```
