import re
from pathlib import Path
from typing import List


def extract_video_number(video_filename: str) -> str:
    """
    Extrae el número del vídeo.

    Ejemplos:
      MVI_0788_VIS_OB.avi -> 0788
      MVI_0804_VIS_OB.avi -> 0804
    """
    m = re.search(r"MVI[_-]?(\d{4})", Path(video_filename).name)
    if not m:
        raise ValueError(f"No se pudo extraer el número del vídeo desde: {video_filename}")
    return m.group(1)


def get_videos_to_process(
    data_dir: Path,
    input_video_files: List[str],
    video_glob: str,
) -> List[Path]:
    """
    Construye la lista de vídeos a procesar:
    - Si input_video_files tiene elementos: usa esa lista dentro de data_dir
    - Si no: usa video_glob dentro de data_dir
    """
    if len(input_video_files) > 0:
        vids = [(data_dir / f).resolve() for f in input_video_files]
    else:
        vids = sorted(data_dir.glob(video_glob))

    vids = [p for p in vids if p.exists()]
    if len(vids) == 0:
        raise FileNotFoundError(
            "No hay vídeos para procesar. Revisa data_dir y input_video_files / video_glob."
        )

    return vids


def get_calib_dir_for_video(video_path: Path, calib_root: Path) -> Path:
    """
    Devuelve la carpeta de calibración para ese vídeo:
      calib_root / <NUM>_calib
    """
    num = extract_video_number(video_path.name)
    calib_dir = (calib_root / f"{num}_calib").resolve()

    if not calib_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de calibración esperada: {calib_dir}")

    return calib_dir