"""Benchmark del pipeline completo con SPEECHLIB_PROFILE=1.

Uso:
    python benchmark_pipeline.py [--audio PATH] [--skip-enhance]

Default: examples/obama_zach.wav (~6 min, 2 speakers)
"""
import os
import sys
import time
import shutil
from pathlib import Path

os.environ["SPEECHLIB_PROFILE"] = "1"

from speechlib.core_analysis import core_analysis


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--audio", default="examples/obama_zach.wav")
    p.add_argument("--skip-enhance", action="store_true")
    p.add_argument("--voices", default=None)
    args = p.parse_args()

    audio = Path(args.audio)
    if not audio.exists():
        print(f"Audio not found: {audio}")
        sys.exit(1)

    # Limpiar cache para medir desde cero
    artifacts = audio.parent / f".{audio.stem.strip()}"
    if artifacts.exists():
        shutil.rmtree(artifacts)
        print(f"Cleaned cache: {artifacts}")

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        print("WARNING: HF_TOKEN not set, diarization will fail")

    print(f"\nBenchmark: {audio.name}")
    print(f"Skip enhance: {args.skip_enhance}")
    print(f"Voices: {args.voices or 'none'}")
    print()

    start = time.perf_counter()
    core_analysis(
        file_name=str(audio),
        voices_folder=args.voices,
        log_folder=None,
        language="es",
        skip_enhance=args.skip_enhance,
    )
    total = time.perf_counter() - start
    print(f"\nTotal wall time: {total:.1f}s")


if __name__ == "__main__":
    main()
