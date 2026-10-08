#!/usr/bin/env bash
# Только собственный стенд; whisper-asr/model-proxy/vLLM не управляются.
set -euo pipefail
# Скрипт не читает команды с клавиатуры; Ctrl+C по-прежнему доставляется сигналом.
exec </dev/null
if [[ $# -lt 1 || ! -d "$1" ]]; then
    echo 'Использование: bash benchmark/run.sh /путь/к/записям [параметры прогона]' >&2
    exit 2
fi
BENCH_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export BENCH_AUDIO_DIR="$(cd -- "$1" && pwd -P)"
shift
BENCH_GIGAAM_ONLY=0
BENCH_FORWARDED_ARGS=()
for bench_arg in "$@"; do
    if [[ "$bench_arg" == '--gigaam-only' ]]; then
        BENCH_GIGAAM_ONLY=1
    else
        BENCH_FORWARDED_ARGS+=("$bench_arg")
    fi
done
set -- "${BENCH_FORWARDED_ARGS[@]}"
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
export BENCH_DOWNLOAD_NETWORK="${BENCH_DOWNLOAD_NETWORK:-host}"
case "$BENCH_DOWNLOAD_NETWORK" in
    host|bridge) ;;
    *) echo 'BENCH_DOWNLOAD_NETWORK должен быть host или bridge' >&2; exit 2 ;;
esac
export BENCH_OLLAMA_DOWNLOAD_PORT="${BENCH_OLLAMA_DOWNLOAD_PORT:-11435}"
[[ "$BENCH_OLLAMA_DOWNLOAD_PORT" =~ ^[0-9]{1,5}$ ]] && \
    (( 10#$BENCH_OLLAMA_DOWNLOAD_PORT >= 1024 && 10#$BENCH_OLLAMA_DOWNLOAD_PORT <= 65535 )) || {
    echo 'BENCH_OLLAMA_DOWNLOAD_PORT должен быть целым числом от 1024 до 65535' >&2; exit 2;
}
export BENCH_OLLAMA_DOWNLOAD_PORT=$((10#$BENCH_OLLAMA_DOWNLOAD_PORT))
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
BENCH_FIRST_LINE_CONTAINER="speech-comparison-first-line-$BENCH_RUN_ID"
BENCH_PREFETCH_CONTAINER="speech-comparison-prefetch-$BENCH_RUN_ID"
BENCH_PREPARE_CONTAINER="speech-comparison-audio-prepare-$BENCH_RUN_ID"
BENCH_CLIENT_CONTAINER="speech-comparison-client-$BENCH_RUN_ID"
BENCH_WHISPER_DOWNLOAD_CONTAINER="speech-comparison-whisper-download-$BENCH_RUN_ID"
BENCH_ARTIFACTS_CONTAINER="speech-comparison-artifacts-$BENCH_RUN_ID"
if (( ! BENCH_GIGAAM_ONLY )); then
    mkdir -p "$BENCH_CACHE/whisper"
fi
mkdir -p "$BENCH_RUN_OUT/логи"
# Запись журнала не должна ждать вывода в SSH/VS Code.
# Основной процесс пишет прямо в файл; отдельный читатель показывает его в терминале.
exec 8>&1
exec >> "$BENCH_RUN_OUT/логи/запуск.log" 2>&1
timeout --foreground --kill-after=3 0 tail --follow=descriptor --sleep-interval=0.1 \
    --pid="$$" -n +1 "$BENCH_RUN_OUT/логи/запуск.log" >&8 2>&8 9>&- &
BENCH_CONSOLE_PID=$!
exec 8>&-
BENCH_CLIENT_READY=0
BENCH_ARTIFACTS_IMAGE=speech-comparison:4.0.0-whisper-standalone
BENCH_COMPLETED=0
BENCH_WHISPER_STARTED=0
BENCH_OLLAMA_STARTED=0
BENCH_DOWNLOAD_STARTED=0
BENCH_MONITOR_PID=''
BENCH_TASK_LOG_PID=''
BENCH_TASK_CONTAINER=''
BENCH_TASK_LOG_FILE=''
bench_compose() {
    docker compose --project-name speech-comparison --project-directory "$BENCH_PROJECT_DIR" \
        -f "$BENCH_PROJECT_DIR/compose.benchmark.yml" "$@"
}
stop_task_log() {
    if [[ -n "$BENCH_TASK_LOG_PID" ]]; then
        kill "$BENCH_TASK_LOG_PID" 2>/dev/null || true
        wait "$BENCH_TASK_LOG_PID" 2>/dev/null || true
        BENCH_TASK_LOG_PID=''
    fi
}
save_task_log() {
    if [[ -n "$BENCH_TASK_CONTAINER" && -n "$BENCH_TASK_LOG_FILE" ]]; then
        timeout --foreground --kill-after=3 30 docker logs "$BENCH_TASK_CONTAINER" > "$BENCH_TASK_LOG_FILE" 2>&1 || return 1
    fi
}
run_task() {
    local limit="$1" container="$2" task_log="$3" exit_code wait_error=0
    shift 3
    BENCH_TASK_CONTAINER="$container"
    BENCH_TASK_LOG_FILE="$BENCH_RUN_OUT/логи/$task_log"
    echo "Запуск задачи $container; управление по статусу контейнера, логи в терминале."
    # Compose только создаёт/запускает задачу. Завершение читаем через Docker wait.
    timeout --foreground 120 docker compose --project-name speech-comparison --project-directory "$BENCH_PROJECT_DIR" \
        -f "$BENCH_PROJECT_DIR/compose.benchmark.yml" run --detach -T --interactive=false \
        --no-deps --name "$container" "$@"
    # SIGTERM у Docker CLI может не завершить заблокированное чтение/вывод.
    # После остановки задачи даём читателю 3 с, затем завершаем только этот CLI.
    # 0 отключает общий дедлайн: логи идут всё время работы контейнера.
    timeout --foreground --kill-after=3 0 docker logs --follow "$container" &
    BENCH_TASK_LOG_PID=$!
    # limit=0 — весь корпус без общего дедлайна; таймаут есть у каждого ASR-запроса.
    if exit_code=$(timeout --foreground "$limit" docker wait "$container"); then
        :
    else
        wait_error=$?
    fi
    stop_task_log
    if (( wait_error == 0 )); then
        echo "Docker сообщил о завершении задачи $container: код $exit_code; поток логов остановлен."
    fi
    if ! save_task_log; then
        echo "Не удалось сохранить отдельный лог $container; общий журнал сохранён."
        (( wait_error != 0 )) || wait_error=1
    fi
    if (( wait_error != 0 )); then
        echo "Ожидание задачи $container не завершено: код $wait_error; останавливаем стенд."
        return "$wait_error"
    fi
    [[ "$exit_code" =~ ^[0-9]{1,3}$ ]] && (( 10#$exit_code <= 255 )) || {
        echo "Docker вернул некорректный код завершения задачи $container: $exit_code"
        return 1
    }
    exit_code=$((10#$exit_code))
    echo "Задача $container завершена: код $exit_code."
    BENCH_TASK_CONTAINER=''
    BENCH_TASK_LOG_FILE=''
    return "$exit_code"
}
stop_test_whisper() {
    if (( BENCH_WHISPER_STARTED )); then
        if ! bench_compose stop whisper-bench; then
            bench_compose logs --no-color --since "$BENCH_STARTED_AT" whisper-bench > "$BENCH_RUN_OUT/логи/whisper-сервер.log" 2>&1 || true
            return 1
        fi
        bench_compose logs --no-color --since "$BENCH_STARTED_AT" whisper-bench > "$BENCH_RUN_OUT/логи/whisper-сервер.log" 2>&1 || true
        BENCH_WHISPER_STARTED=0
        echo 'Тестовый Whisper остановлен; контейнер сохранён, GPU-память освобождается' | tee "$BENCH_RUN_OUT/логи/тестовый-whisper-остановлен.txt"
    fi
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
    if (( bench_exit_code == 0 && ! BENCH_COMPLETED )); then
        bench_exit_code=1
        echo 'Прогон прерван до завершения всех этапов; код 0 заменён на 1.'
    fi
    stop_task_log
    if [[ -n "$BENCH_MONITOR_PID" ]]; then
        kill "$BENCH_MONITOR_PID" 2>/dev/null || true
        wait "$BENCH_MONITOR_PID" 2>/dev/null || true
    fi
    docker stop --time 10 "$BENCH_GIGA_CONTAINER" >/dev/null 2>&1 || true
    docker stop --time 10 "$BENCH_FIRST_LINE_CONTAINER" >/dev/null 2>&1 || true
    docker stop --time 10 "$BENCH_PREFETCH_CONTAINER" >/dev/null 2>&1 || true
    if (( BENCH_GIGAAM_ONLY )); then
        docker stop --time 10 "$BENCH_PREPARE_CONTAINER" >/dev/null 2>&1 || true
    fi
    if (( ! BENCH_GIGAAM_ONLY )); then
        docker stop --time 10 "$BENCH_CLIENT_CONTAINER" >/dev/null 2>&1 || true
        docker stop --time 10 "$BENCH_WHISPER_DOWNLOAD_CONTAINER" >/dev/null 2>&1 || true
    fi
    save_task_log || true
    stop_test_whisper || true
    if (( BENCH_OLLAMA_STARTED )); then
        bench_compose stop ollama || true
        bench_compose logs --no-color --since "$BENCH_STARTED_AT" ollama > "$BENCH_RUN_OUT/логи/ollama.log" 2>&1 || true
    fi
    if (( BENCH_DOWNLOAD_STARTED )); then
        bench_compose logs --no-color --since "$BENCH_STARTED_AT" ollama-download > "$BENCH_RUN_OUT/логи/загрузка-ollama.log" 2>&1 || true
        bench_compose stop ollama-download || true
    fi
    # Останавливаем только контейнеры текущего прогона, сохраняя их и сети.
    echo 'Автоматическое удаление отключено: контейнеры, сети, кеши и результаты сохранены.'
    echo "Завершение: код $bench_exit_code. Продовые контейнеры не управляются."
    if (( BENCH_CLIENT_READY )); then
        # Отчёт не требует сети/GPU; его завершённый контейнер также сохраняется.
        if ! docker run --name "$BENCH_ARTIFACTS_CONTAINER" --network none \
            --memory 2g --cpus 1 --user "$BENCH_UID:$BENCH_GID" --env TZ=Europe/Moscow \
            --env NVIDIA_VISIBLE_DEVICES=void \
            --volume "$BENCH_OUT:/results" --entrypoint python \
            "$BENCH_ARTIFACTS_IMAGE" -m benchmark.artifacts \
            "/results/$BENCH_RUN_ID" --exit-code "$bench_exit_code"; then
            echo 'Не удалось завершить отчёт контейнером; сохраняем архив журналов на хосте'
            archive_logs_on_host || true
            (( bench_exit_code != 0 )) || bench_exit_code=1
        fi
    else
        archive_logs_on_host || true
    fi
    rm -rf -- "$BENCH_RUN_OUT/временные"
    # Даём читателю показать последние строки; зависший терминал не удерживает выход.
    sleep 0.2
    kill "$BENCH_CONSOLE_PID" 2>/dev/null || true
    wait "$BENCH_CONSOLE_PID" 2>/dev/null || true
    exit "$bench_exit_code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'echo "Получен SIGHUP: сессия закрыта, прогон прерывается."; exit 129' HUP
gpu_monitor() {
    echo 'timestamp, uuid, utilization_percent, used_mib, free_mib, total_mib, power_w, temperature_c, phase' > "$BENCH_RUN_OUT/логи/gpu.csv"
    while true; do
        local sample phase free reason
        free=''
        reason=''
        phase="$(cat "$BENCH_RUN_OUT/логи/этап.txt" 2>/dev/null || true)"
        if sample=$(timeout 10 nvidia-smi -i "$BENCH_GPU" --query-gpu=timestamp,uuid,utilization.gpu,memory.used,memory.free,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits 2>>"$BENCH_RUN_OUT/логи/gpu-ошибки.log"); then
            echo "$sample, $phase" >> "$BENCH_RUN_OUT/логи/gpu.csv"
            free=$(awk -F, '{gsub(/ /,"",$5); print $5}' <<< "$sample")
        fi
        if [[ "$phase" == 'GigaAM' || "$phase" == 'GigaAM первая линия' || "$phase" == 'Whisper' ]]; then
            if [[ ! "$free" =~ ^[0-9]+$ ]]; then
                reason='Не удалось проверить свободную память GPU; останавливаем только стенд'
            elif (( free < BENCH_GPU_RESERVE_MIB )); then
                reason="Резерв GPU нарушен: свободно $free МиБ, минимум $BENCH_GPU_RESERVE_MIB. Останавливаем только стенд."
            fi
            if [[ -n "$reason" ]]; then
                echo "$reason" | tee "$BENCH_RUN_OUT/логи/остановка-по-памяти.txt"
                if [[ "$phase" == 'Whisper' ]]; then
                    bench_compose stop whisper-bench >/dev/null 2>&1 || true
                elif [[ "$phase" == 'GigaAM первая линия' ]]; then
                    docker stop --time 5 "$BENCH_FIRST_LINE_CONTAINER" >/dev/null 2>&1 || true
                else
                    docker stop --time 5 "$BENCH_GIGA_CONTAINER" >/dev/null 2>&1 || true
                    bench_compose stop ollama >/dev/null 2>&1 || true
                fi
                return
            fi
        fi
        sleep 2
    done
}
check_capacity() {
    local free_gpu util free_ram docker_root free_disk directory
    free_gpu=$(timeout 10 nvidia-smi -i "$BENCH_GPU" --query-gpu=memory.free --format=csv,noheader,nounits) || return 1
    util=$(timeout 10 nvidia-smi -i "$BENCH_GPU" --query-gpu=utilization.gpu --format=csv,noheader,nounits) || return 1
    free_ram=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)
    echo "Проверка ресурсов: GPU свободно $free_gpu МиБ, загрузка $util%, RAM $free_ram МиБ."
    [[ "$free_gpu" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ && "$free_ram" =~ ^[0-9]+$ ]] || return 1
    (( free_gpu >= BENCH_MIN_FREE && util <= BENCH_GPU_MAX_UTIL && free_ram >= BENCH_MIN_RAM )) || return 1
    docker_root=$(timeout 30 docker info --format '{{.DockerRootDir}}') || return 1
    [[ -d "$docker_root" ]] || return 1
    for directory in "$BENCH_OUT" "$BENCH_CACHE" "$docker_root"; do
        free_disk=$(timeout 30 df -Pm "$directory" | awk 'NR==2 {print $4}') || return 1
        echo "Диск $directory: свободно $free_disk МиБ; минимум $BENCH_MIN_DISK МиБ."
        [[ "$free_disk" =~ ^[0-9]+$ ]] && (( free_disk >= BENCH_MIN_DISK )) || return 1
    done
}
if (( BENCH_GIGAAM_ONLY )); then
    echo "Прогон $BENCH_RUN_ID: только полный цикл GigaAM; GPU $BENCH_GPU."
    echo 'Whisper не запускается; его предыдущие результаты остаются в прежней папке.'
else
    echo "Прогон $BENCH_RUN_ID: собственный Whisper → полный GigaAM → GigaAM первая линия; GPU $BENCH_GPU."
    echo 'Первая линия: только VAD и ASR, без CAM++, эмоций и GigaChat; тот же смешанный корпус, тот же моно WAV.'
    echo 'Адрес Whisper: только whisper-bench:9000 внутри стенда. API и ключи прода не используются.'
fi
echo "Сеть сборки образов: $BENCH_BUILD_NETWORK."
echo "Сеть загрузки весов: $BENCH_DOWNLOAD_NETWORK; загрузчики без GPU и аудиозаписей."
echo "Ollama для загрузки: только 127.0.0.1:$BENCH_OLLAMA_DOWNLOAD_PORT; обработка аудио — во внутренних сетях."
echo 'Сборка образов: Docker Compose, подробный вывод.'
echo "Допустимая загрузка GPU при проверке ресурсов: $BENCH_GPU_MAX_UTIL%."
if (( BENCH_GPU_MAX_UTIL > 10 )); then
    echo 'Допускается рабочая нагрузка на общей GPU. Времена зависят от других сервисов; проверки памяти сохраняются.'
fi
if ! check_capacity; then
    echo "Стенд не запускается: нужно GPU ≥ $BENCH_MIN_FREE МиБ, загрузка ≤ $BENCH_GPU_MAX_UTIL%, RAM ≥ $BENCH_MIN_RAM МиБ, диск ≥ $BENCH_MIN_DISK МиБ. Обработка аудио не начиналась."
    exit 42
fi
docker compose version
docker buildx version
if (( BENCH_GIGAAM_ONLY )); then
    bench_running_whisper=$(bench_compose ps --status running --quiet whisper-bench)
    if [[ -n "$bench_running_whisper" ]]; then
        echo 'Тестовый Whisper ещё работает. Сначала проверьте и остановите его контейнер в проекте speech-comparison; GigaAM не запускается.'
        exit 42
    fi
    echo 'Подготовка GigaAM' > "$BENCH_RUN_OUT/логи/этап.txt"
else
    echo 'Подготовка Whisper' > "$BENCH_RUN_OUT/логи/этап.txt"
fi
# В режиме GigaAM образ нужен только для общего декодера и отчёта, без Whisper ASR.
bench_compose --progress plain build whisper-client
BENCH_CLIENT_READY=1
(
    # Завершение монитора не должно вызывать cleanup всего прогона второй раз.
    trap - EXIT
    trap 'exit 0' TERM INT
    gpu_monitor
) &
BENCH_MONITOR_PID=$!
if (( BENCH_GIGAAM_ONLY )); then
    run_task 0 "$BENCH_PREPARE_CONTAINER" подготовка-аудио.log audio-prepare "$@" \
        --phase gigaam-prepare --audio-dir /recordings --out "/results/$BENCH_RUN_ID"
else
    run_task 3600 "$BENCH_WHISPER_DOWNLOAD_CONTAINER" загрузка-whisper.log whisper-download
    echo 'Загрузка Whisper завершена; проверяем ресурсы перед запуском GPU-сервера.'
    if ! check_capacity; then
        echo 'Ресурсы изменились за время подготовки; тестовый Whisper не запускается'
        exit 42
    fi
    echo 'Whisper' > "$BENCH_RUN_OUT/логи/этап.txt"
    BENCH_WHISPER_STARTED=1
    echo 'Запускаем тестовый Whisper на GPU; ожидание готовности до 600 секунд.'
    bench_compose up -d --wait --wait-timeout 600 whisper-bench
    [[ ! -f "$BENCH_RUN_OUT/логи/остановка-по-памяти.txt" ]] || exit 42
    run_task 0 "$BENCH_CLIENT_CONTAINER" whisper-клиент.log whisper-client "$@" \
        --include-first-line --phase whisper-api --audio-dir /recordings --out "/results/$BENCH_RUN_ID"
    stop_test_whisper
    echo 'Whisper завершён' > "$BENCH_RUN_OUT/логи/этап.txt"
fi
[[ ! -f "$BENCH_RUN_OUT/логи/остановка-по-памяти.txt" ]] || exit 42
if ! check_capacity; then
    echo "GigaAM не запускается: нужно GPU ≥ $BENCH_MIN_FREE МиБ, загрузка ≤ $BENCH_GPU_MAX_UTIL%, RAM ≥ $BENCH_MIN_RAM МиБ. Уже сохранённые результаты остаются в папках прогонов."
    exit 42
fi
echo 'Загрузка моделей' > "$BENCH_RUN_OUT/логи/этап.txt"
bench_compose --progress plain build compare
run_task 3600 "$BENCH_PREFETCH_CONTAINER" загрузка-gigaam.log prefetch
BENCH_DOWNLOAD_STARTED=1
bench_compose up -d --wait --wait-timeout 60 ollama-download
timeout 7200 docker compose --project-name speech-comparison --project-directory "$BENCH_PROJECT_DIR" \
    -f "$BENCH_PROJECT_DIR/compose.benchmark.yml" exec --interactive=false -T ollama-download sh -c \
    'exec ollama pull "$LLM_MODEL"' \
    2>&1 | tee -i "$BENCH_RUN_OUT/логи/загрузка-llm.log"
bench_compose stop ollama-download
if ! check_capacity; then
    echo 'Нагрузка изменилась за время подготовки; GigaAM не запускается'
    exit 42
fi
echo 'GigaAM' > "$BENCH_RUN_OUT/логи/этап.txt"
BENCH_OLLAMA_STARTED=1
bench_compose up -d ollama
[[ ! -f "$BENCH_RUN_OUT/логи/остановка-по-памяти.txt" ]] || exit 42
BENCH_GIGAAM_PHASE=gigaam
if (( BENCH_GIGAAM_ONLY )); then
    BENCH_GIGAAM_PHASE=gigaam-only
fi
run_task 0 "$BENCH_GIGA_CONTAINER" gigaam-контейнер.log compare "$@" --phase "$BENCH_GIGAAM_PHASE" --audio-dir /recordings --out "/results/$BENCH_RUN_ID"
[[ ! -f "$BENCH_RUN_OUT/логи/остановка-по-памяти.txt" ]] || exit 42
if (( ! BENCH_GIGAAM_ONLY )); then
    echo 'Полный GigaAM завершён' > "$BENCH_RUN_OUT/логи/этап.txt"
    # Следующий режим не должен делить GPU с собственной LLM стенда.
    bench_compose stop ollama
    bench_compose logs --no-color --since "$BENCH_STARTED_AT" ollama > "$BENCH_RUN_OUT/логи/ollama.log" 2>&1
    BENCH_OLLAMA_STARTED=0
    echo 'Тестовая Ollama остановлена перед первой линией' > "$BENCH_RUN_OUT/логи/тестовая-ollama-остановлена.txt"
    if ! check_capacity; then
        echo 'Первая линия не запускается: ресурсы изменились; результаты Whisper и полного GigaAM сохранены.'
        exit 42
    fi
    echo 'GigaAM первая линия' > "$BENCH_RUN_OUT/логи/этап.txt"
    run_task 0 "$BENCH_FIRST_LINE_CONTAINER" первая-линия-контейнер.log first-line "$@" \
        --phase gigaam-first-line --audio-dir /recordings --out "/results/$BENCH_RUN_ID"
    [[ ! -f "$BENCH_RUN_OUT/логи/остановка-по-памяти.txt" ]] || exit 42
fi
BENCH_COMPLETED=1
echo "Результаты: $BENCH_RUN_OUT/отчёт.html"
