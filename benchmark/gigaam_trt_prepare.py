"""Экспорт GigaAM в ONNX и сборка TensorRT-движка энкодера; вне замеров."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import shutil
import subprocess
import time

import os

from benchmark.compare import setup_logging
from benchmark.trt_common import save_json
from benchmark.tone_trt_prepare import TRT_IMAGE, TRTEX, gpu_snapshot, sha256, trt_version

log = logging.getLogger("подготовка-gigaam-trt")
GIGAAM_REF = "7447938d791c4f3e643386ee22c33777004293a5"
SYSTEM, LABEL = "gigaam_trt", "GigaAM TensorRT"
MODEL, MODEL_NAME = "gigaam_encoder", "v3_e2e_rnnt"
FEATURES, HIDDEN = 64, 768
# Границы профиля движка в кадрах признаков (10 мс): 0,16 с … 32 с.
MIN_FRAMES, OPT_FRAMES, MAX_FRAMES = 16, 800, 3200
VAD_PROVIDERS = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider"}
TRT_STAGES = {"features": "Звуковые признаки", "encoder": "Энкодер и обмен с Triton", "decoder": "Преобразование в слова"}


def vad_provider() -> str:
    name = os.environ.get("BENCH_VAD_PROVIDER", "cpu").strip().lower()
    if name not in VAD_PROVIDERS:
        raise ValueError("BENCH_VAD_PROVIDER должен быть cpu или cuda")
    return name
PARTS = ("encoder.onnx", "decoder.onnx", "joint.onnx", "preprocessor.pt", "tokenizer.model")


def artifact_names() -> list[str]:
    return [f"{MODEL_NAME}.yaml", *(f"{MODEL_NAME}_{part}" for part in PARTS)]


def export(cache: Path, weights: Path) -> None:
    """ONNX-части модели и состояние её экстрактора признаков. CPU, без аудио."""
    import importlib.metadata

    import gigaam
    import torch

    folder = cache / "artifacts"
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    log.info("Загрузка весов %s на CPU — вне замеров; аудиозаписи этому контейнеру не подключены", MODEL_NAME)
    model = gigaam.load_model(MODEL_NAME, fp16_encoder=False, use_flash=False, device="cpu", download_root=str(weights))
    checkpoint, tokenizer = weights / f"{MODEL_NAME}.ckpt", weights / f"{MODEL_NAME}_tokenizer.model"
    source = {"checkpoint_sha256": sha256(checkpoint), "tokenizer_sha256": sha256(tokenizer), "gigaam_ref": GIGAAM_REF,
              "gigaam": importlib.metadata.version("gigaam"), "torch": torch.__version__}
    manifest_path = cache / "артефакты.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    reused = previous.get("source") == source and all(
        (folder / name).is_file() and sha256(folder / name) == previous.get("files", {}).get(name, {}).get("sha256")
        for name in artifact_names())
    if reused:
        log.info("ONNX-части уже в кеше для этих весов; SHA256 проверен")
    else:
        log.info("Экспорт энкодера, декодера и joint в ONNX fp32")
        model.to_onnx(str(folder))
        # Окно и мел-фильтры лежат в весах и отличаются от создаваемых заново: без них текст другой.
        torch.save(model.preprocessor.state_dict(), folder / f"{MODEL_NAME}_preprocessor.pt")
        shutil.copyfile(tokenizer, folder / f"{MODEL_NAME}_tokenizer.model")
    save_json(manifest_path, {
        "model": MODEL_NAME, "source": source, "reused": reused,
        "blank_id": model.decoding.blank_id, "max_symbols": model.decoding.max_symbols,
        "pred_hidden": model.cfg.head.decoder.pred_hidden, "pred_rnn_layers": model.cfg.head.decoder.pred_rnn_layers,
        "exported_at": datetime.now().astimezone().isoformat(), "preparation_seconds": time.perf_counter() - started,
        "files": {name: {"size_bytes": (folder / name).stat().st_size, "sha256": sha256(folder / name)}
                  for name in artifact_names()}})
    log.info("Экспорт подготовлен за %.1f с", time.perf_counter() - started)


def model_config(instances: int = 1) -> str:
    # Один запрос — один фрагмент речи; батчей и CPU-инстансов нет.
    return f'''name: "{MODEL}"
platform: "tensorrt_plan"
max_batch_size: 0
input [
  {{ name: "audio_signal" data_type: TYPE_FP32 dims: [1, {FEATURES}, -1] }},
  {{ name: "length" data_type: TYPE_INT64 dims: [1] }}
]
output [
  {{ name: "encoded" data_type: TYPE_FP32 dims: [1, {HIDDEN}, -1] }},
  {{ name: "encoded_len" data_type: TYPE_INT32 dims: [1] }}
]
instance_group [{{ count: {instances} kind: KIND_GPU gpus: [0] }}]
'''


def compile_engine(cache: Path, output: Path, workspace_mib: int, timeout: int, instances: int = 1) -> None:
    manifest_path = cache / "артефакты.json"
    if not manifest_path.is_file():
        raise RuntimeError("Нет экспортированной модели; сначала нужен экспорт")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    name = f"{MODEL_NAME}_encoder.onnx"
    model = cache / "artifacts" / name
    if not model.is_file() or sha256(model) != manifest["files"][name]["sha256"]:
        raise RuntimeError("ONNX-энкодер отсутствует или повреждён; повторите экспорт")
    gpu = gpu_snapshot()
    version = trt_version()
    repository = cache / "repository" / MODEL
    directory = repository / "1"
    directory.mkdir(parents=True, exist_ok=True)
    engine, temporary = directory / "model.plan", directory / "model.plan.part"
    shapes = lambda frames: f"audio_signal:1x{FEATURES}x{frames},length:1"
    command = [TRTEX, f"--onnx={model}", f"--minShapes={shapes(MIN_FRAMES)}", f"--optShapes={shapes(OPT_FRAMES)}",
        f"--maxShapes={shapes(MAX_FRAMES)}", "--fp16", "--skipInference",
        f"--memPoolSize=workspace:{workspace_mib}", f"--saveEngine={temporary}"]
    fingerprint = {"gpu": {key: gpu[key] for key in ("name", "uuid", "driver_version")},
        "trtexec_version": version, "triton_image": TRT_IMAGE, "model_sha256": manifest["files"][name]["sha256"],
        "command": command, "config_sha256": hashlib.sha256(model_config().encode()).hexdigest()}
    metadata_path = cache / "движок.json"
    previous = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    reused = (engine.is_file() and previous.get("fingerprint") == fingerprint
              and sha256(engine) == previous.get("engine_sha256"))
    started = time.perf_counter()
    if reused:
        log.info("TensorRT-движок уже подготовлен для этой GPU и версии; SHA256 проверен")
    else:
        log.info("Сборка TensorRT на %s; fp16, фрагменты %.2f–%.0f с, workspace=%d МиБ — вне замеров",
                 gpu["name"], MIN_FRAMES / 100, MAX_FRAMES / 100, workspace_mib)
        log.info("Команда: %s", " ".join(command))
        try:
            subprocess.run(command, check=True, timeout=timeout)
            if not temporary.is_file() or not temporary.stat().st_size:
                raise RuntimeError("trtexec завершился без готового движка")
            temporary.replace(engine)
        finally:
            temporary.unlink(missing_ok=True)
    # Движок один на все копии модели: их число меняет только конфигурацию сервера.
    (repository / "config.pbtxt").write_text(model_config(instances))
    metadata = {"fingerprint": fingerprint, "instances": instances, "engine_sha256": sha256(engine),
        "engine_size_bytes": engine.stat().st_size, "reused": reused,
        "preparation_seconds": time.perf_counter() - started,
        "prepared_at": datetime.now().astimezone().isoformat(), "gpu_after": gpu_snapshot(),
        "included_in_measurements": False}
    save_json(metadata_path, metadata)
    save_json(output / "tensorRT-подготовка.json", metadata)
    log.info("TensorRT подготовлен за %.3f с; GPU после подготовки: %s", metadata["preparation_seconds"], metadata["gpu_after"])


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("export", "compile"))
    cli.add_argument("--cache", type=Path, default=Path("/cache"))
    cli.add_argument("--weights", type=Path, default=Path("/weights"))
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--workspace-mib", type=int, default=4096)
    cli.add_argument("--timeout", type=int, default=3600)
    cli.add_argument("--instances", type=int, default=1, help="Копий модели в Triton")
    args = cli.parse_args()
    if not 512 <= args.workspace_mib <= 8192 or args.timeout <= 0 or not 1 <= args.instances <= 16:
        cli.error("Workspace 512–8192 МиБ; время сборки положительное; копий модели от 1 до 16")
    args.out.mkdir(parents=True, exist_ok=True)
    setup_logging(args.out)
    if args.phase == "export":
        export(args.cache, args.weights)
    else:
        compile_engine(args.cache, args.out, args.workspace_mib, args.timeout, args.instances)


if __name__ == "__main__":
    main()
