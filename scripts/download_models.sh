#!/bin/sh
# Скачивает модели распознавания речи и голосов в указанную папку (по умолчанию ./models).
# Используется при сборке образа; можно запустить и вручную для работы без Docker.
#
#   sh scripts/download_models.sh [папка]
#
# Источник — официальные релизы проекта sherpa-onnx на GitHub. Если GitHub из вашей сети
# недоступен, укажите зеркало: MODELS_BASE_URL=https://... sh scripts/download_models.sh
set -eu

DEST="${1:-./models}"
BASE="${MODELS_BASE_URL:-https://github.com/k2-fsa/sherpa-onnx/releases/download}"

ASR_NAME="sherpa-onnx-nemo-transducer-punct-giga-am-v3-russian-2025-12-16"
ASR_SHA="f9620a0099019c6afcee26525ef9ed3297fa50dd5691c1902af0c948fc1a470b"
SPK_NAME="3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
SPK_SHA="aa3cfc16963a10586a9393f5035d6d6b57e98d358b347f80c2a30bf4f00ceba2"

fetch() {  # fetch <url> <файл> <sha256>
    echo ">> $1"
    curl -fL --retry 5 --retry-delay 3 --connect-timeout 20 -o "$2" "$1"
    echo "$3  $2" | sha256sum -c -
}

mkdir -p "$DEST/asr"
cd "$DEST"

# GigaAM-v3 e2e RNN-T с пунктуацией (int8 ONNX), лицензия MIT
if [ "${DOWNLOAD_ASR:-1}" = 1 ] && [ ! -f asr/encoder.int8.onnx ]; then
    fetch "$BASE/asr-models/$ASR_NAME.tar.bz2" asr.tar.bz2 "$ASR_SHA"
    tar -xjf asr.tar.bz2
    for f in encoder.int8.onnx decoder.onnx joiner.onnx tokens.txt LICENSE; do
        mv "$ASR_NAME/$f" "asr/$f"
    done
    rm -rf asr.tar.bz2 "$ASR_NAME"
fi

# 3D-Speaker CAM++ — вектор голоса для разделения собеседников, лицензия Apache-2.0
if [ ! -f speaker.onnx ]; then
    fetch "$BASE/speaker-recongition-models/$SPK_NAME" speaker.onnx "$SPK_SHA"
fi

echo "Модели готовы: $(pwd)"
ls -lh asr speaker.onnx
