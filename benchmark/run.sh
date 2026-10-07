#!/usr/bin/env bash
# Только собственный стенд; whisper-asr/model-proxy/vLLM не управляются.
set -euo pipefail
if [[ $# -lt 1 || ! -d "$1" ]]; then
    echo 'Использование: bash benchmark/run.sh /путь/к/записям [параметры прогона]' >&2
    exit 2
fi
BENCH_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export BENCH_AUDIO_DIR="$(cd -- "$1" && pwd -P)"
shift
exec 9>"$BENCH_PROJECT_DIR/.benchmark.lock"
flock -n 9 || { echo 'Другой прогон этого стенда уже выполняется' >&2; exit 2; }
export BENCH_OUT="${BENCH_OUT:-$BENCH_PROJECT_DIR/benchmark-results}"
export BENCH_CACHE="${BENCH_CACHE:-$BENCH_PROJECT_DIR/benchmark-cache}"
export BENCH_THREADS="${BENCH_THREADS:-4}"
export BENCH_GPU="${BENCH_GPU:-0}"
export BENCH_BUILD_NETWORK="${BENCH_BUILD_NETWORK:-default}"
case "$BENCH_BUILD_NETWORK" in
    default|host) ;;
    *) echo 'BENCH_BUILD_NETWORK должен быть default или host' >&2; exit 2 ;;
esac
export BENCH_CPUSET="${BENCH_CPUSET:-0-3}"
export BENCH_UID="$(id -u)"
export BENCH_GID="$(id -g)"
BENCH_MIN_FREE="${BENCH_GPU_MIN_FREE_MIB:-16384}"
export BENCH_GPU_MAX_UTIL="${BENCH_GPU_MAX_UTIL:-10}"
[[ "$BENCH_GPU_MAX_UTIL" =~ ^[0-9]{1,3}$ ]] && (( 10#$BENCH_GPU_MAX_UTIL <= 100 )) || {
    echo 'BENCH_GPU_MAX_UTIL должен быть целым числом от 0 до 100' >&2; exit 2;
}
export BENCH_GPU_MAX_UTIL=$((10#$BENCH_GPU_MAX_UTIL))
export BENCH_GPU_RESERVE_MIB="${BENCH_GPU_RESERVE_MIB:-4096}"
BENCH_MIN_RAM="${BENCH_MIN_RAM_MIB:-20480}"
BENCH_MIN_DISK="${BENCH_MIN_DISK_MIB:-32768}"
export BENCH_GPU_MIN_FREE_MIB="$BENCH_MIN_FREE"
export BENCH_MIN_RAM_MIB="$BENCH_MIN_RAM"
for bench_number in "$BENCH_MIN_FREE" "$BENCH_GPU_RESERVE_MIB" "$BENCH_MIN_RAM" "$BENCH_MIN_DISK" "$BENCH_THREADS"; do
    [[ "$bench_number" =~ ^[0-9]+$ ]] || { echo 'Порог памяти/число потоков должны быть целыми числами' >&2; exit 2; }
done
BENCH_RUN_ID="$(TZ=Europe/Moscow date +%Y%m%d-%H%M%S)-$$"
BENCH_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
mkdir -p -- "$BENCH_OUT" "$BENCH_CACHE"
export BENCH_OUT="$(cd -- "$BENCH_OUT" && pwd -P)"
export BENCH_CACHE="$(cd -- "$BENCH_CACHE" && pwd -P)"
case "$BENCH_OUT/" in "$BENCH_AUDIO_DIR/"*) echo 'Отчёты должны быть вне папки записей' >&2; exit 2 ;; esac
case "$BENCH_CACHE/" in "$BENCH_AUDIO_DIR/"*) echo 'Кеш должен быть вне папки записей' >&2; exit 2 ;; esac
BENCH_RUN_OUT="$BENCH_OUT/$BENCH_RUN_ID"
BENCH_GIGA_CONTAINER="speech-comparison-gigaam-$BENCH_RUN_ID"
BENCH_PREFETCH_CONTAINER="speech-comparison-prefetch-$BENCH_RUN_ID"
mkdir -p "$BENCH_RUN_OUT/логи"
exec > >(tee -i -a "$BENCH_RUN_OUT/логи/запуск.log") 2>&1
BENCH_CLIENT_READY=0
BENCH_OLLAMA_STARTED=0
BENCH_DOWNLOAD_STARTED=0
BENCH_MONITOR_PID=''
bench_compose() {
    docker compose --project-name speech-comparison --project-directory "$BENCH_PROJECT_DIR" \
        -f "$BENCH_PROJECT_DIR/compose.benchmark.yml" "$@"
}
archive_logs_on_host() {
    local temporary_archive
    # Запись внутри исходной папки меняет её во время чтения tar.
    temporary_archive=$(mktemp "$BENCH_OUT/.диагностика-$BENCH_RUN_ID-XXXXXX.tar.part") || return 1
    if tar --exclude='./диагностика.tar.gz' --exclude='./диагностика.tar.part' --exclude='./временные' \
        -czf "$temporary_archive" -C "$BENCH_RUN_OUT" . && \
        mv -f -- "$temporary_archive" "$BENCH_RUN_OUT/диагностика.tar.gz"; then
        return 0
    fi
    rm -f -- "$temporary_archive"
    return 1
}
cleanup() {
    local bench_exit_code=$?
    trap - EXIT
    if [[ -n "$BENCH_MONITOR_PID" ]]; then
        kill "$BENCH_MONITOR_PID" 2>/dev/null || true
        wait "$BENCH_MONITOR_PID" 2>/dev/null || true
    fi
    docker stop --time 10 "$BENCH_GIGA_CONTAINER" >/dev/null 2>&1 || true
    docker stop --time 10 "$BENCH_PREFETCH_CONTAINER" >/dev/null 2>&1 || true
    if (( BENCH_OLLAMA_STARTED )); then
        bench_compose logs --no-color --since "$BENCH_STARTED_AT" ollama > "$BENCH_RUN_OUT/логи/ollama.log" 2>&1 || true
        bench_compose stop ollama || true
        bench_compose rm -f ollama || true
    fi
    if (( BENCH_DOWNLOAD_STARTED )); then
        bench_compose logs --no-color --since "$BENCH_STARTED_AT" ollama-download > "$BENCH_RUN_OUT/логи/загрузка-ollama.log" 2>&1 || true
        bench_compose stop ollama-download || true
        bench_compose rm -f ollama-download || true
    fi
    echo "Завершение: код $bench_exit_code. Рабочий Whisper не управляется."
    if (( BENCH_CLIENT_READY )); then
        if ! bench_compose run --rm --no-deps --entrypoint python whisper-client -m benchmark.artifacts \
            "/results/$BENCH_RUN_ID" --exit-code "$bench_exit_code"; then
            echo 'Не удалось завершить отчёт контейнером; сохраняем архив журналов на хосте'
            archive_logs_on_host || true
            (( bench_exit_code != 0 )) || bench_exit_code=1
        fi
    else
        archive_logs_on_host || true
    fi
    rm -rf -- "$BENCH_RUN_OUT/временные"
    return "$bench_exit_code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
gpu_monitor() {
    echo 'timestamp, uuid, utilization_percent, used_mib, free_mib, total_mib, power_w, temperature_c, phase' > "$BENCH_RUN_OUT/логи/gpu.csv"
    while true; do
        local sample phase free
        phase="$(cat "$BENCH_RUN_OUT/логи/этап.txt" 2>/dev/null || true)"
        if sample=$(nvidia-smi -i "$BENCH_GPU" --query-gpu=timestamp,uuid,utilization.gpu,memory.used,memory.free,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits 2>>"$BENCH_RUN_OUT/логи/gpu-ошибки.log"); then
            echo "$sample, $phase" >> "$BENCH_RUN_OUT/логи/gpu.csv"
            free=$(awk -F, '{gsub(/ /,"",$5); print $5}' <<< "$sample")
            if [[ "$phase" == 'GigaAM' && "$free" =~ ^[0-9]+$ ]] && (( free < BENCH_GPU_RESERVE_MIB )); then
                echo "Резерв GPU нарушен: свободно $free МиБ, минимум $BENCH_GPU_RESERVE_MIB. Останавливаем только стенд." | tee "$BENCH_RUN_OUT/логи/остановка-по-памяти.txt"
                docker stop --time 5 "$BENCH_GIGA_CONTAINER" >/dev/null 2>&1 || true
                bench_compose stop ollama >/dev/null 2>&1 || true
                return
            fi
        elif [[ "$phase" == 'GigaAM' ]]; then
            echo 'Не удалось проверить GPU; останавливаем только стенд' > "$BENCH_RUN_OUT/логи/остановка-по-памяти.txt"
            docker stop --time 5 "$BENCH_GIGA_CONTAINER" >/dev/null 2>&1 || true
            bench_compose stop ollama >/dev/null 2>&1 || true
            return
        fi
        sleep 2
    done
}
check_capacity() {
    local free_gpu util free_ram docker_root free_disk directory
    free_gpu=$(nvidia-smi -i "$BENCH_GPU" --query-gpu=memory.free --format=csv,noheader,nounits)
    util=$(nvidia-smi -i "$BENCH_GPU" --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    free_ram=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
    echo "Перед GigaAM: GPU свободно $free_gpu МиБ, загрузка $util%, RAM $free_ram МиБ."
    [[ "$free_gpu" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ && "$free_ram" =~ ^[0-9]+$ ]] || return 1
    (( free_gpu >= BENCH_MIN_FREE && util <= BENCH_GPU_MAX_UTIL && free_ram >= BENCH_MIN_RAM )) || return 1
    docker_root=$(docker info --format '{{.DockerRootDir}}') || return 1
    [[ -d "$docker_root" ]] || return 1
    for directory in "$BENCH_OUT" "$BENCH_CACHE" "$docker_root"; do
        free_disk=$(df -Pm "$directory" | awk 'NR==2 {print $4}')
        echo "Диск $directory: свободно $free_disk МиБ; минимум $BENCH_MIN_DISK МиБ."
        [[ "$free_disk" =~ ^[0-9]+$ ]] && (( free_disk >= BENCH_MIN_DISK )) || return 1
    done
}
echo "Прогон $BENCH_RUN_ID: весь корпус Whisper API → весь корпус GigaAM; GPU $BENCH_GPU."
echo "Сеть сборки образов: $BENCH_BUILD_NETWORK."
echo 'Сборка образов: Docker Compose, подробный вывод.'
echo "Допустимая загрузка GPU при проверке ресурсов: $BENCH_GPU_MAX_UTIL%."
if (( BENCH_GPU_MAX_UTIL > 10 )); then
    echo 'Допускается рабочая нагрузка на общей GPU. Времена зависят от других сервисов; проверки памяти сохраняются.'
fi
if ! check_capacity; then
    echo "Стенд не запускается: нужно GPU ≥ $BENCH_MIN_FREE МиБ, загрузка ≤ $BENCH_GPU_MAX_UTIL%, RAM ≥ $BENCH_MIN_RAM МиБ, диск ≥ $BENCH_MIN_DISK МиБ. Запросов Whisper не было."
    exit 42
fi
docker compose version
docker buildx version
bench_compose --progress plain build whisper-client
BENCH_CLIENT_READY=1
echo 'Whisper API' > "$BENCH_RUN_OUT/логи/этап.txt"
gpu_monitor &
BENCH_MONITOR_PID=$!
bench_compose run --rm --no-deps whisper-client --phase whisper-api --audio-dir /recordings --out "/results/$BENCH_RUN_ID" "$@"
if ! check_capacity; then
    echo "GigaAM не запускается: нужно GPU ≥ $BENCH_MIN_FREE МиБ, загрузка ≤ $BENCH_GPU_MAX_UTIL%, RAM ≥ $BENCH_MIN_RAM МиБ. Результаты Whisper сохранены."
    exit 42
fi
echo 'Загрузка моделей' > "$BENCH_RUN_OUT/логи/этап.txt"
bench_compose --progress plain build compare
timeout 3600 docker compose --project-name speech-comparison --project-directory "$BENCH_PROJECT_DIR" \
    -f "$BENCH_PROJECT_DIR/compose.benchmark.yml" run --rm --no-deps --name "$BENCH_PREFETCH_CONTAINER" prefetch
BENCH_DOWNLOAD_STARTED=1
bench_compose up -d ollama-download
timeout 7200 docker compose --project-name speech-comparison --project-directory "$BENCH_PROJECT_DIR" \
    -f "$BENCH_PROJECT_DIR/compose.benchmark.yml" exec -T ollama-download sh -c \
    'for i in $(seq 1 60); do if ollama list >/dev/null 2>&1; then exec ollama pull "$LLM_MODEL"; fi; sleep 1; done; exit 1'
bench_compose stop ollama-download
if ! check_capacity; then
    echo 'Нагрузка изменилась за время подготовки; GigaAM не запускается'
    exit 42
fi
echo 'GigaAM' > "$BENCH_RUN_OUT/логи/этап.txt"
BENCH_OLLAMA_STARTED=1
bench_compose up -d ollama
[[ ! -f "$BENCH_RUN_OUT/логи/остановка-по-памяти.txt" ]] || exit 42
bench_compose run --rm --no-deps --name "$BENCH_GIGA_CONTAINER" compare --phase gigaam --audio-dir /recordings --out "/results/$BENCH_RUN_ID" "$@"
[[ ! -f "$BENCH_RUN_OUT/логи/остановка-по-памяти.txt" ]] || exit 42
echo "Результаты: $BENCH_RUN_OUT/отчёт.html"
