"""Перенос образа: реквизиты прокси, доступ контейнера и платформа архива."""
import io
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import tarfile

import pytest

from benchmark import fetch_triton_image as transfer


def test_proxy_credentials_are_redacted_in_raw_and_decoded_forms():
    proxy = "http://private-user:private%21pass@proxy.example:3128"
    text = f"{proxy}; private-user; private%21pass; private!pass"
    safe = transfer.redact(text, proxy)
    assert "private" not in safe and safe.count("[скрыто]") == 4


def test_downloader_never_receives_gpu_audio_docker_socket_or_proxy_arguments(tmp_path):
    command = transfer.container_command("own-download", tmp_path,
        ["copy", "docker://" + transfer.TRITON_SOURCE, "docker-archive:/out/image.tar:" + transfer.TRITON_TAG],
        tmp_path / "temporary-layers")
    assert "--gpus" not in command and "docker.sock" not in " ".join(command)
    assert "/recordings" not in " ".join(command)
    assert command[command.index("--log-driver") + 1] == "none"
    assert [command[i+1] for i, value in enumerate(command) if value == "--mount"] == [
        f"type=bind,src={tmp_path},dst=/out",
        f"type=bind,src={tmp_path / 'temporary-layers'},dst=/var/tmp"]
    assert command[command.index("--pull") + 1] == "never"
    assert "--rm" not in command and "--privileged" not in command
    assert "@sha256:" in transfer.SKOPEO_IMAGE and "@sha256:" in transfer.TRITON_SOURCE
    assert [command[i+1] for i, value in enumerate(command) if value == "--env"] == ["HOME=/tmp"]


def test_proxy_is_passed_by_stdin_and_not_written_to_console_or_log(capsys):
    proxy = "http://private-user:private-pass@proxy.example:3128"
    command = [sys.executable, "-c", "import sys; print(sys.stdin.readline().strip()); print('private-pass')"]
    log = io.StringIO()
    transfer.run_container(command, proxy, log)
    assert "private" not in log.getvalue() and "private" not in capsys.readouterr().out
    assert log.getvalue().count("[скрыто]") == 2


@pytest.mark.parametrize("exit_code", [0, 1])
def test_import_only_loads_archive_and_does_not_hide_docker_failure(monkeypatch, tmp_path, exit_code):
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, exit_code, stdout="Результат импорта\n")
    monkeypatch.setattr(transfer.subprocess, "run", run)
    target = tmp_path / "checked.tar"
    log = io.StringIO()
    if exit_code:
        with pytest.raises(subprocess.CalledProcessError):
            transfer.load_archive(target, log)
    else:
        transfer.load_archive(target, log)
    assert commands == [["docker", "image", "load", "--input", str(target)]]
    assert "Результат импорта" in log.getvalue()


@pytest.mark.parametrize("tag,arch,accepted", [
    (transfer.TRITON_TAG, "amd64", True),
    ("other/image:latest", "amd64", False),
    (transfer.TRITON_TAG, "arm64", False),
])
def test_archive_requires_exact_official_tag_and_linux_amd64(tmp_path, tag, arch, accepted):
    path = tmp_path / "image.tar"
    manifest = [{"Config": "config.json", "RepoTags": [tag], "Layers": []}]
    config = {"architecture": arch, "os": "linux"}
    with tarfile.open(path, "w") as archive:
        for name, data in (("manifest.json", manifest), ("config.json", config)):
            content = json.dumps(data).encode()
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    if accepted:
        transfer.verify_archive(path)
    else:
        with pytest.raises(RuntimeError):
            transfer.verify_archive(path)
    assert not (tmp_path / "config.json").exists()


@pytest.mark.skipif(os.environ.get("TRITON_TRANSFER_DOCKER_TEST") != "1",
    reason="Явный локальный тест Skopeo без сети, GPU, аудио и моделей")
def test_real_skopeo_archive_copy_needs_writable_var_tmp(tmp_path):
    source = tmp_path / "source"
    temporary = tmp_path / "temporary-layers"
    source.mkdir(); temporary.mkdir()
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        content = b"local temporary-directory regression check"
        info = tarfile.TarInfo("test.txt")
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
    uncompressed = stream.getvalue()
    layer = gzip.compress(uncompressed, mtime=0)
    config = json.dumps({"architecture": "amd64", "os": "linux", "config": {},
        "rootfs": {"type": "layers", "diff_ids": ["sha256:" + hashlib.sha256(uncompressed).hexdigest()]},
        "history": [{"created_by": "local-test-only"}]}).encode()
    def descriptor(data, media_type):
        digest = hashlib.sha256(data).hexdigest()
        (source / digest).write_bytes(data)
        return {"mediaType": media_type, "digest": "sha256:" + digest, "size": len(data)}
    manifest = {"schemaVersion": 2, "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
        "config": descriptor(config, "application/vnd.docker.container.image.v1+json"),
        "layers": [descriptor(layer, "application/vnd.docker.image.rootfs.diff.tar.gzip")]}
    (source / "manifest.json").write_text(json.dumps(manifest))
    for fixed in (False, True):
        name = f"speech-comparison-image-tempcheck-{os.getpid()}-{int(fixed)}"
        target = f"test-{int(fixed)}.tar"
        command = transfer.container_command(name, tmp_path,
            ["copy", "dir:/out/source", f"docker-archive:/out/{target}:local-transfer-check:test"], temporary)
        command[command.index("--network") + 1] = "none"
        if not fixed:
            position = command.index(f"type=bind,src={temporary},dst=/var/tmp")
            del command[position-1:position+1]
        result = subprocess.run(command, input="\n", capture_output=True, text=True, timeout=30)
        if not fixed:
            assert result.returncode != 0 and "read-only file system" in result.stderr
            assert "/var/tmp/" in result.stderr
        else:
            assert result.returncode == 0, result.stdout + result.stderr
            with tarfile.open(tmp_path / target, "r:") as archive:
                loaded = json.load(archive.extractfile("manifest.json"))
                assert loaded[0]["RepoTags"] == ["docker.io/library/local-transfer-check:test"]
                assert json.load(archive.extractfile(loaded[0]["Config"]))["architecture"] == "amd64"
