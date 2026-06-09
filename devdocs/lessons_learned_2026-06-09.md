# Lessons Learned — Session 2026-06-09

## What is speechlib

A Python library (repo `speechlib`, package `speechlib==1.1.16`) that does
speaker diarization, transcription and speaker recognition on audio. It runs in
its own uv-managed venv (`C:\workspace\dev\speechlib\.venv`) and is invoked as a
subprocess by a separate consumer (the `biz_long_term_memory` transcribe stage)
via `<venv>\python.exe -m speechlib ...`.

## Glossary

- **self-package**: the project's own importable package (`speechlib/`), as
  opposed to its third-party dependencies (torch, faster-whisper, …).
- **`[build-system]`**: the PEP 517 table in `pyproject.toml` declaring the build
  backend. Its presence/absence decides whether a tool treats the project as an
  installable package or as a bare dependency manifest.
- **virtual project (uv)**: how recent uv treats a project with no `[build-system]`
  — it resolves and installs the project's *dependencies* but does NOT build or
  install the project itself into the venv.

## Session intent

The downstream `biz_long_term_memory` transcribe stage was failing with 0 VTTs.
The captured error was
`...speechlib\.venv\Scripts\python.exe: No module named speechlib`. Goal: root-
cause why a fully-provisioned speechlib venv could not import its own package,
and fix it so transcription works again.

## What worked

### 1. Declaring `[build-system]` so uv installs the self-package
- **Problem**: `import speechlib` raised `ModuleNotFoundError: No module named
  'speechlib'` inside the project's own venv, even though all heavy dependencies
  were present (257 packages in `site-packages`: torch 2.12.0+cu126,
  faster-whisper 1.2.1, accelerate, pyannote, …). The package directory
  `speechlib/__init__.py` existed on disk but no `speechlib` dist-info or editable
  `.pth` was in `site-packages`. `uv sync` reported `Resolved 163 / Checked 128`
  and installed nothing — it considered the project already in sync.
- **Solution**: `pyproject.toml` had `[tool.setuptools.packages.find]` but **no
  `[build-system]` table at all**. Added the standard setuptools backend:
  ```toml
  [build-system]
  requires = ["setuptools>=61"]
  build-backend = "setuptools.build_meta"
  ```
- **Result**: `uv sync` then ran `Building speechlib @ file:///.../speechlib`,
  `Built speechlib`, `Installed 1 package`, `+ speechlib==1.1.16 (from file://...)`.
  `import speechlib` → `OK C:\workspace\dev\speechlib\speechlib\__init__.py`. The
  downstream `transcribe-session` then produced the VTT (1/1).

## Key technical findings

1. **Without a `[build-system]` table, recent uv treats the project as a
   "virtual" project: it installs the dependencies but never builds/installs the
   self-package.** This is why `site-packages` had every third-party dep yet no
   `speechlib`. Evidence: adding `[build-system]` flipped `uv sync` from
   "Checked 128, installed nothing" to "Built speechlib, + speechlib==1.1.16",
   2026-06-09.
2. **`git log -S 'build-system' -- pyproject.toml` returned nothing — the table
   was never present.** The package used to be importable because an older
   provisioning (pre-uv mamba env, or an older uv that packaged without an
   explicit backend) had placed it in the venv. A later clean `uv sync` with
   current uv silently dropped the self-package. So this is a latent regression
   exposed by re-provisioning, not a removed line.
3. **The failure mode is silent at sync time.** `uv sync` exits 0 and prints a
   reassuring "Resolved/Checked" summary while leaving the project unimportable.
   The only signal is an `ImportError` at run time. A `[build-system]` table is
   the cheap guarantee that the self-package is actually built into the venv.

## Structural changes learned this session

```
pyproject.toml
  BEFORE:  [tool.setuptools.packages.find]   ← backend config present
           (no [build-system] table)         ← but no backend DECLARED
           => uv installs deps only, NOT speechlib itself
              => import speechlib -> ModuleNotFoundError

  AFTER:   [build-system]
             requires = ["setuptools>=61"]
             build-backend = "setuptools.build_meta"
           [tool.setuptools.packages.find]
           => uv builds + installs speechlib==1.1.16 into the venv
```

The lesson is the pairing: `[tool.setuptools.*]` configures a backend that is
only used if `[build-system]` actually names it. Configuration without
declaration is inert under uv.
