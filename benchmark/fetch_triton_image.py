"""Скачать официальный Triton через прокси, не меняя настройки службы Docker.

На хосте используется только стандартная библиотека Python. Сеть и
скачивание слоёв выполняет отдельный официальный контейнер Skopeo.
"""
import argparse
from datetime import datetime
import getpass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from urllib.parse import unquote, urlsplit


TRITON_TAG = "nvcr.io/nvidia/tritonserver:25.06-py3"
TRITON_SOURCE = "nvcr.io/nvidia/tritonserver@sha256:75bcfa5b0043898ece3e603c17a5bbbb1c9bddc390563db24312ef59d83735e5"
SKOPEO_IMAGE = "quay.io/skopeo/stable@sha256:e5d5d2815b94d74767d40be7111647c3e9cdc84406a638ffe1c7ea1f286f52a3"
PROXY_ENTRYPOINT = (
    'IFS= read -r HTTPS_PROXY || exit 2; '
    'HTTP_PROXY="$HTTPS_PROXY"; export HTTPS_PROXY HTTP_PROXY; '
    'exec skopeo "$@"'
)


def redact(text, proxy):
    parsed = urlsplit(proxy)
    secrets = [proxy, parsed.username, parsed.password]
    secrets += [unquote(value) for value in secrets if value]
    for secret in sorted(set(filter(None, secrets)), key=len, reverse=True):
        text = text.replace(secret, "[скрыто]")
    return text


def container_command(name, folder, arguments, temporary_dir):
    return ["docker", "run", "--interactive", "--init", "--pull", "never", "--name", name,
        "--label", "org.temptest.task=triton-image-download",
        "--network", "host", "--memory", "1g", "--cpus", "2",
        "--log-driver", "none", "--read-only", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges:true",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--tmpfs", "/tmp:mode=1777,size=512m", "--env", "HOME=/tmp",
        "--mount", f"type=bind,src={folder},dst=/out",
        # Skopeo хранит большие промежуточные слои в /var/tmp. Это диск
        # внутри своей папки передачи, а не ограниченный tmpfs или RAM.
        "--mount", f"type=bind,src={temporary_dir},dst=/var/tmp",
        "--entrypoint", "/bin/sh", SKOPEO_IMAGE,
        "-c", PROXY_ENTRYPOINT, "image-transfer",
        "--override-os", "linux", "--override-arch", "amd64", *arguments]


def run_container(command, proxy, log):
    # Адрес с паролем передаётся только по stdin: его нет в аргументах,
    # окружении Docker-контейнера, журнале или сохраняемых метаданных.
    process = subprocess.Popen(command, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        process.stdin.write(proxy + "\n")
        process.stdin.close()
        for line in process.stdout:
            safe = redact(line, proxy)
            print(safe, end="", flush=True)
            log.write(safe)
            log.flush()
        code = process.wait()
    except BaseException:
        # Прерывание касается только созданного загрузчика. ID сверяется
        # по имени, проверяется собственная метка; других контейнеров нет.
        name = command[command.index("--name") + 1]
        try:
            inspected = subprocess.run(["docker", "inspect", "--format",
                '{{.Id}} {{index .Config.Labels "org.temptest.task"}}', name],
                capture_output=True, text=True, timeout=10)
            fields = inspected.stdout.strip().split()
            if (inspected.returncode == 0 and len(fields) == 2
                    and fields[1] == "triton-image-download" and len(fields[0]) == 64
                    and all(character in "0123456789abcdef" for character in fields[0])):
                subprocess.run(["docker", "stop", "--time", "5", fields[0]],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
        except (OSError, subprocess.TimeoutExpired):
            pass
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    finally:
        process.stdout.close()
    if code:
        raise RuntimeError(f"Контейнер загрузки завершился с кодом {code}")


def verify_archive(path):
    # Ничего не извлекаем на диск. Проверяем тег и платформу docker-archive.
    with tarfile.open(path, "r:") as archive:
        manifest_file = archive.extractfile("manifest.json")
        manifest = json.loads(manifest_file.read(4 * 1024 * 1024))
        if len(manifest) != 1 or manifest[0].get("RepoTags") != [TRITON_TAG]:
            raise RuntimeError("В архиве неожиданный тег или несколько образов")
        config_file = archive.extractfile(manifest[0]["Config"])
        config = json.loads(config_file.read(4 * 1024 * 1024))
        if config.get("architecture") != "amd64" or config.get("os") != "linux":
            raise RuntimeError("Образ должен иметь платформу linux/amd64")


def load_archive(path, log):
    # Импортируется проверенный образ; операции с контейнерами отсутствуют.
    print("Загрузка готового образа в локальный Docker...", flush=True)
    result = subprocess.run(["docker", "image", "load", "--input", str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout, end="", flush=True)
    log.write(result.stdout)
    log.flush()
    result.check_returncode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Проверить доступ, без скачивания слоёв Triton")
    parser.add_argument("--load", action="store_true", help="После проверки архива загрузить образ в Docker этой машины")
    args = parser.parse_args()
    if args.check_only and args.load:
        parser.error("--check-only и --load нельзя указывать вместе")
    root = Path(__file__).resolve().parents[1]
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    folder = root / "benchmark-cache" / "triton-image-transfer" / run_id
    folder.mkdir(parents=True)
    if not args.check_only and shutil.disk_usage(folder).free < 40 * 1024**3:
        raise RuntimeError("Для образа и архива нужно не менее 40 ГиБ свободного диска")
    proxy = getpass.getpass("Вставь полный адрес HTTP-прокси (ввод скрыт): ")
    parsed = urlsplit(proxy)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or "\n" in proxy or "\r" in proxy:
        raise ValueError("Нужен полный адрес HTTP/HTTPS-прокси")
    if subprocess.run(["docker", "image", "inspect", SKOPEO_IMAGE],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        print("Подготовка официального контейнера-загрузчика — без GPU.", flush=True)
        subprocess.run(["docker", "pull", SKOPEO_IMAGE], check=True)
    name = f"speech-comparison-image-download-{run_id}"
    print(f"Контейнер: {name}\nПапка: {folder}", flush=True)
    print("Работающие контейнеры и настройки Docker не меняются. Контейнер-загрузчик после завершения сохраняется.", flush=True)
    partial = folder / "triton-25.06-py3.tar.part"
    if args.check_only:
        arguments = ["inspect", "--format", "{{.Architecture}} {{.Digest}}", "docker://" + TRITON_SOURCE]
    else:
        arguments = ["copy", "--retry-times", "3", "--digestfile", "/out/исходный-digest.txt",
            "docker://" + TRITON_SOURCE,
            "docker-archive:/out/triton-25.06-py3.tar.part:" + TRITON_TAG]
    with (folder / "загрузка.log").open("w", encoding="utf-8") as log:
        with tempfile.TemporaryDirectory(prefix=".слои-", dir=folder) as temporary_dir:
            run_container(container_command(name, folder, arguments, temporary_dir), proxy, log)
    if args.check_only:
        print("Проверка доступа завершена; слои Triton не скачивались.")
        return
    print("Проверка платформы и контрольной суммы архива...", flush=True)
    verify_archive(partial)
    digest = hashlib.sha256()
    with partial.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    target = partial.with_suffix("")
    partial.rename(target)
    (folder / "triton-25.06-py3.tar.sha256").write_text(
        f"{digest.hexdigest()}  {target.name}\n", encoding="utf-8")
    (folder / "образ.json").write_text(json.dumps({"source": TRITON_SOURCE,
        "tag": TRITON_TAG, "platform": "linux/amd64", "archive_sha256": digest.hexdigest(),
        "container": name, "skopeo_image": SKOPEO_IMAGE}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Архив готов: {target}", flush=True)
    if args.load:
        with (folder / "загрузка.log").open("a", encoding="utf-8") as log:
            load_archive(target, log)
        print(f"Готово: {TRITON_TAG} загружен в Docker. Можно запускать обычную команду теста T-one TensorRT.")
    else:
        print("Передай архив и файл .sha256 на сервер.")
        print("На сервере: sha256sum -c triton-25.06-py3.tar.sha256 && docker image load -i triton-25.06-py3.tar")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nЗагрузка прервана. Исходные записи и работающие контейнеры не изменены.", file=sys.stderr)
        sys.exit(130)
    except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError, tarfile.TarError) as error:
        # Не выводим исключение с возможными прокси-реквизитами.
        print(f"Загрузка не завершена ({type(error).__name__}). См. журнал в benchmark-cache/triton-image-transfer.", file=sys.stderr)
        sys.exit(1)
