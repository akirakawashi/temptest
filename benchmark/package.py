"""Упаковывает исходники файлового стенда; аудио и модели передаются отдельно."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parent.parent
PREFIX = "speech-comparison"
EXCLUDED = {"__pycache__", ".pytest_cache", ".venv", ".env"}


def source_files() -> list[Path]:
    paths = [ROOT / name for name in (
        "Dockerfile.benchmark", "Dockerfile.whisper-client", "compose.benchmark.yml", "requirements.txt", ".dockerignore",
        "Dockerfile.tone-benchmark", "compose.tone-benchmark.yml",
        "Dockerfile.tone-trt-benchmark", "compose.tone-trt-benchmark.yml",
        "scripts/download_models.sh",
    )]
    for folder in ("app", "benchmark"):
        for path in (ROOT / folder).rglob("*"):
            if path.is_file() and not (set(path.relative_to(ROOT).parts) & EXCLUDED):
                if path.suffix not in {".pyc", ".pyo"}:
                    paths.append(path)
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Нужен обычный файл исходников: {path}")
    return sorted(paths)


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", type=Path, default=ROOT.parent / "speech-comparison-server.tar.gz")
    args = cli.parse_args()
    output = args.out.resolve()
    if output.exists():
        cli.error("Архив уже существует; укажите новое имя через --out")
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = source_files()
    manifest = {
        "format_version": 1,
        "description": "Отдельный Whisper large-v3 и полный цикл GigaAM: CUDA-стенд",
        "audio_included": False,
        "model_cache_included": False,
        "files": [{"path": str(path.relative_to(ROOT)), "size_bytes": path.stat().st_size,
                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths],
    }
    data = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    temporary = output.with_name(output.name + ".part")
    try:
        with tarfile.open(temporary, "w:gz") as archive:
            for path in paths:
                archive.add(path, arcname=f"{PREFIX}/{path.relative_to(ROOT)}", recursive=False)
            info = tarfile.TarInfo(f"{PREFIX}/MANIFEST.json")
            info.size, info.mode = len(data), 0o644
            archive.addfile(info, io.BytesIO(data))
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_name(output.name + ".sha256").write_text(f"{digest}  {output.name}\n", encoding="ascii")
    print(f"Архив: {output}; файлов исходников: {len(paths)}; размер: {output.stat().st_size} байт")


if __name__ == "__main__":
    main()
