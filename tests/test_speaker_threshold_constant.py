"""AT: SPEAKER_SIMILARITY_THRESHOLD constante única en speaker_recognition.

Slice 15: los tests de find_best_speaker fueron eliminados junto con la
funcion (cubierta por _best_match en speechlib/domain/recognition.py y
sus tests en test_domain_recognition.py / test_acceptance_recognition_quality.py).
"""


def test_constant_value_is_050():
    """Threshold subido de 0.45 -> 0.50 tras el fix de Pamela bug:
    el nuevo embedding selection (top-N por duracion) surfaceó un falso
    positivo en Alicanto SPEAKER_02 -> Cristian Ruiz a sim=0.498. Pamela
    (0.71) y Agustin (0.72) siguen pasando holgadamente."""
    from speechlib.speaker_recognition import SPEAKER_SIMILARITY_THRESHOLD
    assert SPEAKER_SIMILARITY_THRESHOLD == 0.50


def test_min_margin_constant_exists():
    """Slice 9: agrega constante para el margen minimo top1 vs top2."""
    from speechlib.speaker_recognition import SPEAKER_SIMILARITY_MIN_MARGIN
    assert SPEAKER_SIMILARITY_MIN_MARGIN == 0.10


def test_relabel_vtt_imports_threshold_from_core():
    """relabel_vtt no debe definir DEFAULT_THRESHOLD localmente; debe importarlo del core."""
    import ast
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] /
           "speechlib" / "tools" / "relabel_vtt.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    # No debe haber asignación literal: DEFAULT_THRESHOLD = 0.40
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "DEFAULT_THRESHOLD":
                    if isinstance(node.value, ast.Constant):
                        raise AssertionError("DEFAULT_THRESHOLD definido como literal en relabel_vtt.py")


def test_diagnose_speaker_imports_threshold_from_core():
    """diagnose_speaker no debe definir THRESHOLD localmente."""
    import ast
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] /
           "speechlib" / "tools" / "diagnose_speaker.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "THRESHOLD":
                    if isinstance(node.value, ast.Constant):
                        raise AssertionError("THRESHOLD definido como literal en diagnose_speaker.py")
