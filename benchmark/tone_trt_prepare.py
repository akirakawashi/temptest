"""Загрузка и сборка T-one TensorRT в контейнерах; стандартная библиотека."""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import time
import urllib.request

from benchmark.compare import setup_logging
from benchmark.tone_run import save_json

log = logging.getLogger("подготовка-t-one-trt")
TONE_REF = "3c5b6c015038173840e62cea99e10cdb1c759116"
HF_REF = "106f3b0b32a9e107eb613312e4ebc61ff3d53926"
ARTIFACTS = {
    "model.onnx": (144193371, "707e4a282d5036304a0b8603ad83945840c3075f53c49911ef88c19bcc5a7d52"),
    "kenlm.bin": (5463477004, "8c31a489a51a6e9236112dacb6bed12f45e8df734057615fa6bf220a5a769a1d"),
}
TRT_IMAGE = "nvcr.io/nvidia/tritonserver:25.06-py3"
TRTEX = "/usr/src/tensorrt/bin/trtexec"


def trt_version() -> str:
    # Версия из той же библиотеки, с которой связан trtexec. Флаг --version
    # поддерживается не всеми выпусками trtexec; Python bindings не нужны.
    library = ctypes.CDLL("libnvinfer.so.10")
    library.getInferLibVersion.argtypes = []
    library.getInferLibVersion.restype = ctypes.c_int
    value = library.getInferLibVersion()
    if value < 100000 or value >= 110000:
        raise RuntimeError(f"Нужен TensorRT 10.x из закреплённого образа; версия {value}")
    return f"{value // 10000}.{value % 10000 // 100}.{value % 100}"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(cache: Path, with_kenlm: bool) -> None:
    folder = cache / "artifacts"
    folder.mkdir(parents=True, exist_ok=True)
    items = {}
    for name in ("model.onnx", "kenlm.bin") if with_kenlm else ("model.onnx",):
        size, digest = ARTIFACTS[name]
        target = folder / name
        log.info("Проверка %s (%d байт) — вне замеров", name, size)
        if target.is_file() and target.stat().st_size == size and sha256(target) == digest:
            log.info("%s уже в кеше; SHA256 проверен", name)
        else:
            temporary = target.with_suffix(target.suffix + ".part")
            url = f"https://huggingface.co/t-tech/T-one/resolve/{HF_REF}/{name}"
            log.info("Скачивание %s; аудиозаписи этому контейнеру не подключены", url)
            started = last = time.monotonic()
            downloaded, checksum = 0, hashlib.sha256()
            try:
                with urllib.request.urlopen(url, timeout=90) as response, temporary.open("wb") as stream:
                    while block := response.read(8 * 1024 ** 2):
                        stream.write(block)
                        checksum.update(block)
                        downloaded += len(block)
                        now = time.monotonic()
                        if now - last >= 5:
                            log.info("%s: %.1f%%, %.0f / %.0f МиБ, %.1f МиБ/с", name,
                                downloaded / size * 100, downloaded / 1024 ** 2, size / 1024 ** 2,
                                downloaded / 1024 ** 2 / (now - started))
                            last = now
                if downloaded != size or checksum.hexdigest() != digest:
                    raise RuntimeError(f"Размер или SHA256 {name} не совпал с закреплённой версией")
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
            log.info("%s скачан за %.1f с; SHA256 проверен", name, time.monotonic() - started)
        items[name] = {"size_bytes": size, "sha256": digest}
    save_json(cache / "артефакты.json", {"tone_ref": TONE_REF, "hf_ref": HF_REF,
        "verified_at": datetime.now().astimezone().isoformat(), "files": items})


def gpu_snapshot() -> dict:
    query = "name,uuid,driver_version,memory.free,memory.total,utilization.gpu"
    completed = subprocess.run(["nvidia-smi", f"--query-gpu={query}",
        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=True)
    values = [part.strip() for part in completed.stdout.strip().split(",")]
    if len(values) != 6:
        raise RuntimeError("nvidia-smi не вернула сведения о единственной выбранной GPU")
    info = dict(zip(("name", "uuid", "driver_version", "free_mib", "total_mib", "utilization_percent"), values))
    for key in ("free_mib", "total_mib", "utilization_percent"):
        info[key] = int(info[key])
    reserve = int(os.environ.get("BENCH_GPU_RESERVE_MIB", "3072"))
    if info["free_mib"] < reserve:
        raise RuntimeError(f"Нарушен резерв GPU: свободно {info['free_mib']} МиБ, нужно {reserve}")
    return info


def model_config() -> str:
    # Только batch=1; задержки накопления батча нет. Никаких CPU-инстансов.
    return '''name: "streaming_acoustic"
platform: "tensorrt_plan"
max_batch_size: 1
input [
  { name: "signal" data_type: TYPE_INT32 dims: [2400, 1] },
  { name: "state" data_type: TYPE_FP16 dims: [219729] }
]
output [
  { name: "logprobs" data_type: TYPE_FP32 dims: [10, 35] },
  { name: "state_next" data_type: TYPE_FP16 dims: [219729] }
]
instance_group [{ count: 1 kind: KIND_GPU gpus: [0] }]
model_warmup [{
  name: "warmup_batch_one" batch_size: 1 count: 10
  inputs: { key: "signal" value: { data_type: TYPE_INT32 dims: [2400, 1] zero_data: true } }
  inputs: { key: "state" value: { data_type: TYPE_FP16 dims: [219729] zero_data: true } }
}]
'''


def compile_engine(cache: Path, output: Path, workspace_mib: int, timeout: int) -> None:
    model = cache / "artifacts/model.onnx"
    if not model.is_file() or sha256(model) != ARTIFACTS["model.onnx"][1]:
        raise RuntimeError("Официальная ONNX-модель отсутствует или повреждена; сначала нужна загрузка")
    gpu = gpu_snapshot()
    version = trt_version()
    repository = cache / "repository/streaming_acoustic"
    directory = repository / "1"
    directory.mkdir(parents=True, exist_ok=True)
    engine, temporary = directory / "model.plan", directory / "model.plan.part"
    command = [TRTEX, f"--onnx={model}",
        "--minShapes=signal:1x2400x1,state:1x219729",
        "--optShapes=signal:1x2400x1,state:1x219729",
        "--maxShapes=signal:1x2400x1,state:1x219729",
        "--builderOptimizationLevel=5", "--stronglyTyped", "--skipInference",
        f"--memPoolSize=workspace:{workspace_mib}", f"--saveEngine={temporary}"]
    fingerprint = {"gpu": {key: gpu[key] for key in ("name", "uuid", "driver_version")},
        "trtexec_version": version, "triton_image": TRT_IMAGE,
        "model_sha256": ARTIFACTS["model.onnx"][1], "command": command,
        "config_sha256": hashlib.sha256(model_config().encode()).hexdigest()}
    metadata_path = cache / "движок.json"
    previous = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    reused = (engine.is_file() and previous.get("fingerprint") == fingerprint
              and sha256(engine) == previous.get("engine_sha256"))
    started = time.perf_counter()
    if reused:
        log.info("TensorRT-движок уже подготовлен для этой GPU и версии; SHA256 проверен")
    else:
        log.info("Сборка TensorRT на %s; batch=1, workspace=%d МиБ — вне замеров", gpu["name"], workspace_mib)
        log.info("Команда: %s", " ".join(command))
        try:
            subprocess.run(command, check=True, timeout=timeout)
            if not temporary.is_file() or not temporary.stat().st_size:
                raise RuntimeError("trtexec завершился без готового движка")
            temporary.replace(engine)
        finally:
            temporary.unlink(missing_ok=True)
    (repository / "config.pbtxt").write_text(model_config())
    metadata = {"fingerprint": fingerprint, "engine_sha256": sha256(engine),
        "engine_size_bytes": engine.stat().st_size, "reused": reused,
        "preparation_seconds": time.perf_counter() - started,
        "prepared_at": datetime.now().astimezone().isoformat(), "gpu_after": gpu_snapshot(),
        "included_in_measurements": False}
    save_json(metadata_path, metadata)
    save_json(output / "tensorRT-подготовка.json", metadata)
    log.info("TensorRT подготовлен за %.3f с; GPU после подготовки: %s", metadata["preparation_seconds"], metadata["gpu_after"])


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("download", "compile"))
    cli.add_argument("--cache", type=Path, default=Path("/cache"))
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--without-kenlm", action="store_true")
    cli.add_argument("--workspace-mib", type=int, default=1024)
    cli.add_argument("--timeout", type=int, default=3600)
    args = cli.parse_args()
    if not 128 <= args.workspace_mib <= 4096 or args.timeout <= 0:
        cli.error("Workspace 128–4096 МиБ; время сборки положительное")
    args.out.mkdir(parents=True, exist_ok=True)
    setup_logging(args.out)
    if args.phase == "download":
        download(args.cache, not args.without_kenlm)
    else:
        compile_engine(args.cache, args.out, args.workspace_mib, args.timeout)


if __name__ == "__main__":
    main()
