from pathlib import Path
import sys


def add_lens_distortion_to_path() -> Path:
    """
    Busca la carpeta third_party/LensDistortionFromLines/python
    subiendo desde este archivo y la añade a sys.path.
    Devuelve la ruta encontrada.
    """
    current_file = Path(__file__).resolve()

    for parent in [current_file.parent] + list(current_file.parents):
        candidate = parent / "third_party" / "LensDistortionFromLines" / "python"
        if candidate.exists() and candidate.is_dir():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return candidate

    raise FileNotFoundError(
        "No se encontró la carpeta 'third_party/LensDistortionFromLines/python'."
    )