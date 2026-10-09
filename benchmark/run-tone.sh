#!/usr/bin/env bash
# Только T-one; никаких команд управления продовыми сервисами.
set -euo pipefail
exec </dev/null
if [[ $# -lt 1 || ! -d "$1" ]]; then
    echo 'Использование: bash benchmark/run-tone.sh /папка/аудио [--reference-run /старый/прогон] [параметры]' >&2
    exit 2
fi
TONE_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export BENCH_AUDIO_DIR="$(cd -- "$1" && pwd -P)"
shift
TONE_ARGS=()
TONE_REFERENCE_ARGS=()
TONE_REFERENCE=''
while (( $# )); do
    case "$1" in
        --reference-run)
            [[ $# -ge 2 && -z "$TONE_REFERENCE" && -f "$2/условия.json" ]] || {
                echo '--reference-run требует папку предыдущего прогона с условия.json' >&2; exit 2;
            }
            TONE_REFERENCE="$(cd -- "$2" && pwd -P)"
            TONE_REFERENCE_ARGS=(--reference-run /reference)
            shift 2 ;;
        --out|--model-dir|--finalize|--out=*|--model-dir=*|--finalize=*)
            echo 'Папку результатов задаёт стенд; модель и завершающий этап фиксированы' >&2; exit 2 ;;
        *) TONE_ARGS+=("$1"); shift ;;
    esac
done
# Общий замок с прежним тестом: два наших прогона одновременно не стартуют.
exec 9>"$TONE_PROJECT_DIR/.benchmark.lock"
flock -n 9 || { echo 'Другой тест этого стенда уже выполняется' >&2; exit 2; }
export BENCH_OUT="${BENCH_OUT:-$TONE_PROJECT_DIR/benchmark-results}"
export BENCH_CACHE="${BENCH_CACHE:-$TONE_PROJECT_DIR/benchmark-cache}"
export BENCH_GPU="${BENCH_GPU:-0}"
export BENCH_THREADS="${BENCH_THREADS:-4}"
export BENCH_CPUSET="${BENCH_CPUSET:-0-3}"
export BENCH_GPU_MIN_FREE_MIB="${BENCH_GPU_MIN_FREE_MIB:-12288}"
export BENCH_GPU_MAX_UTIL="${BENCH_GPU_MAX_UTIL:-10}"
export BENCH_GPU_RESERVE_MIB="${BENCH_GPU_RESERVE_MIB:-3072}"
export BENCH_BUILD_NETWORK="${BENCH_BUILD_NETWORK:-default}"
export BENCH_UID="$(id -u)" BENCH_GID="$(id -g)"
TONE_MIN_RAM="${BENCH_MIN_RAM_MIB:-20480}"
TONE_MIN_DISK="${BENCH_MIN_DISK_MIB:-32768}"
for number in "$BENCH_GPU_MIN_FREE_MIB" "$BENCH_GPU_MAX_UTIL" "$BENCH_GPU_RESERVE_MIB" "$BENCH_THREADS" "$TONE_MIN_RAM" "$TONE_MIN_DISK"; do
    [[ "$number" =~ ^[0-9]+$ ]] || { echo 'Пороги ресурсов и число потоков должны быть целыми числами' >&2; exit 2; }
done
(( BENCH_GPU_MAX_UTIL <= 100 && BENCH_THREADS > 0 )) || exit 2
[[ "$BENCH_BUILD_NETWORK" == default || "$BENCH_BUILD_NETWORK" == host ]] || exit 2
mkdir -p -- "$BENCH_OUT" "$BENCH_CACHE"
export BENCH_OUT="$(cd -- "$BENCH_OUT" && pwd -P)"
export BENCH_CACHE="$(cd -- "$BENCH_CACHE" && pwd -P)"
case "$BENCH_OUT/" in "$BENCH_AUDIO_DIR/"*) echo 'Результаты должны быть вне папки аудио' >&2; exit 2 ;; esac
TONE_RUN_ID="tone-$(TZ=Europe/Moscow date +%Y%m%d-%H%M%S)-$$"
TONE_RUN_OUT="$BENCH_OUT/$TONE_RUN_ID"
export BENCH_REFERENCE_DIR="${TONE_REFERENCE:-$TONE_PROJECT_DIR/benchmark}"
TONE_CONTAINER="speech-comparison-tone-$TONE_RUN_ID"
TONE_ARTIFACTS="speech-comparison-tone-artifacts-$TONE_RUN_ID"
mkdir -p "$TONE_RUN_OUT/логи"
TONE_LOG="$TONE_RUN_OUT/логи/запуск.log"
exec 3>&1 4>&2
exec >>"$TONE_LOG" 2>&1
tail --pid=$$ --follow=descriptor --sleep-interval=0.2 --lines=+1 "$TONE_LOG" >&3 2>&4 9>&- &
TONE_CONSOLE_PID=$!
TONE_LOG_PID='' TONE_MONITOR_PID='' TONE_STARTED=0 TONE_IMAGE_READY=0
tone_compose() { docker compose --project-name speech-comparison-tone -f "$TONE_PROJECT_DIR/compose.tone-benchmark.yml" "$@"; }
cleanup() {
    local exit_code=$?
    local finalized=0
    trap - EXIT INT TERM HUP
    [[ -z "$TONE_MONITOR_PID" ]] || kill "$TONE_MONITOR_PID" 2>/dev/null || true
    if (( TONE_STARTED )); then
        timeout 30 docker stop --time 10 "$TONE_CONTAINER" >/dev/null 2>&1 || true
        [[ -z "$TONE_LOG_PID" ]] || kill "$TONE_LOG_PID" 2>/dev/null || true
        timeout 30 docker logs "$TONE_CONTAINER" > "$TONE_RUN_OUT/логи/t-one.log" 2>&1 || true
    fi
    echo "Завершение: код $exit_code. Продовые контейнеры не управляются. Автоматическое удаление отключено."
    if (( TONE_IMAGE_READY )); then
        if timeout --kill-after=5s 120 docker run --name "$TONE_ARTIFACTS" \
            --label com.docker.compose.project=speech-comparison-tone \
            --network none --memory 2g --cpus 1 --user "$BENCH_UID:$BENCH_GID" \
            --env NVIDIA_VISIBLE_DEVICES=void --env TZ=Europe/Moscow \
            --volume "$BENCH_OUT:/results" --entrypoint python \
            speech-comparison:1.0.0-tone-cuda -m benchmark.tone_run \
            --finalize "/results/$TONE_RUN_ID" "$exit_code"; then
            finalized=1
        else
            echo 'Не удалось завершить отчёт контейнером; пробуем стандартную библиотеку Python на сервере.'
            (( exit_code != 0 )) || exit_code=1
        fi
    fi
    if (( ! finalized )); then
        # Работает даже без построенного образа; pip и модели не нужны.
        (cd -- "$TONE_PROJECT_DIR" && PYTHONDONTWRITEBYTECODE=1 python3 -m benchmark.tone_run \
            --finalize "$TONE_RUN_OUT" "$exit_code") || {
            echo 'Архив создать не удалось; полные журналы сохранены в папке прогона.'
            (( exit_code != 0 )) || exit_code=1
        }
    fi
    sleep 0.3
    kill "$TONE_CONSOLE_PID" 2>/dev/null || true
    exit "$exit_code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'echo "SSH-сессия закрылась: прогон прерывается"; exit 129' HUP
check_capacity() {
    local sample free util ram disk docker_root directory
    sample=$(timeout 10 nvidia-smi -i "$BENCH_GPU" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits)
    IFS=, read -r free util <<< "${sample// /}"
    ram=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
    echo "Ресурсы: GPU свободно $free МиБ, загрузка $util%; RAM $ram МиБ."
    [[ "$free" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ && "$ram" =~ ^[0-9]+$ ]] || return 1
    (( free >= BENCH_GPU_MIN_FREE_MIB && util <= BENCH_GPU_MAX_UTIL && ram >= TONE_MIN_RAM )) || return 1
    docker_root=$(timeout 30 docker info --format '{{.DockerRootDir}}')
    for directory in "$BENCH_OUT" "$docker_root"; do
        disk=$(df -Pm "$directory" | awk 'NR==2 {print $4}')
        echo "Диск $directory: свободно $disk МиБ."
        [[ "$disk" =~ ^[0-9]+$ ]] && (( disk >= TONE_MIN_DISK )) || return 1
    done
}
monitor_gpu() {
    echo 'timestamp,uuid,utilization_percent,used_mib,free_mib,total_mib,power_w,temperature_c' > "$TONE_RUN_OUT/логи/gpu.csv"
    while true; do
        local sample free
        sample=$(timeout 10 nvidia-smi -i "$BENCH_GPU" --query-gpu=timestamp,uuid,utilization.gpu,memory.used,memory.free,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits) || sample=''
        echo "$sample" >> "$TONE_RUN_OUT/логи/gpu.csv"
        free=$(awk -F, '{gsub(/ /,"",$5); print $5}' <<< "$sample")
        if [[ ! "$free" =~ ^[0-9]+$ ]] || (( free < BENCH_GPU_RESERVE_MIB )); then
            echo "Резерв GPU нарушен или память недоступна: $free МиБ; минимум $BENCH_GPU_RESERVE_MIB. Останавливаем только T-one." | tee "$TONE_RUN_OUT/логи/остановка-по-памяти.txt"
            timeout 30 docker stop --time 5 "$TONE_CONTAINER" >/dev/null 2>&1 || true
            return
        fi
        sleep 2
    done
}
echo "Отдельный прогон T-one $TONE_RUN_ID; GPU $BENCH_GPU; потоков $BENCH_THREADS; сеть обработки отключена."
echo "Результаты: $TONE_RUN_OUT; исходные записи: $BENCH_AUDIO_DIR"
if ! check_capacity; then echo 'Недостаточно ресурсов для заданных порогов; модели не запускались.'; exit 42; fi
if ! docker image inspect speech-comparison:4.0.0-gigaam-cuda >/dev/null 2>&1; then
    echo 'Готового CUDA-образа прежнего стенда нет; собираем его библиотеки без запуска сервисов.'
    docker compose --project-name speech-comparison-tone -f "$TONE_PROJECT_DIR/compose.benchmark.yml" \
        --progress plain build compare
fi
tone_compose --progress plain build tone
TONE_IMAGE_READY=1
docker image inspect --format '{{json .}}' speech-comparison:1.0.0-tone-cuda > "$TONE_RUN_OUT/логи/образ.json"
if ! check_capacity; then echo 'После сборки недостаточно ресурсов; модели не запускались.'; exit 42; fi
TONE_STARTED=1
tone_compose run --detach --no-deps --name "$TONE_CONTAINER" tone \
    /recordings --out "/results/$TONE_RUN_ID" "${TONE_REFERENCE_ARGS[@]}" "${TONE_ARGS[@]}"
timeout --kill-after=3s 86400 docker logs --follow "$TONE_CONTAINER" 9>&- &
TONE_LOG_PID=$!
monitor_gpu 9>&- &
TONE_MONITOR_PID=$!
while true; do
    state=$(timeout 10 docker inspect --format '{{.State.Running}} {{.State.ExitCode}}' "$TONE_CONTAINER")
    read -r running exit_code <<< "$state"
    if [[ "$running" == false ]]; then
        [[ ! -f "$TONE_RUN_OUT/логи/остановка-по-памяти.txt" ]] || exit_code=42
        exit "$exit_code"
    fi
    sleep 2
done
