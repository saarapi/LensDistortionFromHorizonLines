import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple


def ensure_ffmpeg_available() -> None:
    """
    Comprueba que ffmpeg está disponible en PATH.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("No se encontró 'ffmpeg' en PATH. Instálalo o añade ffmpeg al PATH.")


def run_cmd(cmd: List[str]) -> None:
    """
    Ejecuta un comando y, si falla, lanza un error con stdout/stderr.
    """
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "Error ejecutando comando.\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{e.stdout.decode('utf-8', errors='ignore')}\n"
            f"STDERR:\n{e.stderr.decode('utf-8', errors='ignore')}\n"
        )


def extract_sampled_frames_ffmpeg(
    video_path: Path,
    extracted_dir: Path,
    *,
    every_n: int,
    max_frames: int,
) -> List[Tuple[Path, int, int]]:
    """
    Extrae frames con ffmpeg:
      - 1 de cada every_n frames
      - máximo max_frames imágenes

    Devuelve:
      [(png_path, sample_idx, approx_frame_idx)]

    donde:
      sample_idx = 0..max_frames-1
      approx_frame_idx ≈ sample_idx * every_n
    """
    ensure_ffmpeg_available()
    extracted_dir.mkdir(parents=True, exist_ok=True)

    # Limpiar frames previos
    for f in extracted_dir.glob("sample_*.png"):
        f.unlink()

    out_pattern = str(extracted_dir / "sample_%05d.png")
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vf", f"select='not(mod(n,{int(every_n)}))'",
        "-vsync", "vfr",
        "-frames:v", str(int(max_frames)),
        out_pattern,
    ]
    run_cmd(cmd)

    files = sorted(extracted_dir.glob("sample_*.png"))
    if len(files) == 0:
        raise RuntimeError(f"No se extrajo ningún frame desde {video_path} (ffmpeg).")

    out: List[Tuple[Path, int, int]] = []
    for i, p in enumerate(files[:max_frames]):
        out.append((p, i, i * int(every_n)))

    return out