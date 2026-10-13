#!/usr/bin/env bash
# Только отдельный GigaAM TensorRT: поиск речи, распознавание, паузы; с --full — ещё голоса и эмоции. Ни одной команды управления продом.
set -euo pipefail
exec </dev/null
if [[ $# -lt 1 || ! -d "$1" ]]; then
    echo 'Использование: bash benchmark/run-gigaam-trt.sh /папка/аудио [--full] [--parallel N] [--reference-run /старый/прогон]' >&2
    exit 2
fi
TRT_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export BENCH_AUDIO_DIR="$(cd -- "$1" && pwd -P)"
shift
TRT_ARGS=() TRT_REFERENCE_ARGS=()
TRT_REFERENCE=''
TRT_PARALLEL=1 TRT_FULL=0
TRT_IMAGE='nvcr.io/nvidia/tritonserver:25.06-py3'
while (( $# )); do
    case "$1" in
        --reference-run)
            [[ $# -ge 2 && -z "$TRT_REFERENCE" && -f "$2/условия.json" ]] || {
                echo '--reference-run требует папку предыдущего прогона с условия.json' >&2; exit 2;
            }
            TRT_REFERENCE="$(cd -- "$2" && pwd -P)"
            TRT_REFERENCE_ARGS=(--reference-run /reference)
            shift 2 ;;
        --out|--model-dir|--finalize|--out=*|--model-dir=*|--finalize=*)
            echo 'Папки модели и результатов задаются стендом' >&2; exit 2 ;;
        --parallel)
            [[ $# -ge 2 && "$2" =~ ^[0-9]{1,2}$ ]] && (( 10#$2 >= 1 && 10#$2 <= 32 )) || {
                echo '--parallel требует число одновременных записей от 1 до 32' >&2; exit 2;
            }
            TRT_PARALLEL=$((10#$2)); shift 2 ;;
        --full) TRT_FULL=1; shift ;;
        --gap|--gap=*|--expected-files|--expected-files=*)
            echo "$1 больше нет: обрабатываются все записи папки, пауз между ними нет" >&2; exit 2 ;;
        *) TRT_ARGS+=("$1"); shift ;;
    esac
done
exec 9>"$TRT_PROJECT_DIR/.benchmark.lock"
flock -n 9 || { echo 'Другой тест стенда уже выполняется' >&2; exit 2; }
export BENCH_OUT="${BENCH_OUT:-$TRT_PROJECT_DIR/benchmark-results}"
export BENCH_CACHE="${BENCH_CACHE:-$TRT_PROJECT_DIR/benchmark-cache}"
export BENCH_GPU="${BENCH_GPU:-0}" BENCH_THREADS="${BENCH_THREADS:-4}"
# В полном flow у каждого обработчика своя модель эмоций на GPU. Замер на RTX 4070 SUPER: 2,7 ГиБ памяти
# и 0,9 ГиБ видеопамяти на обработчик; значения ниже взяты с запасом для общей карты.
TRT_STREAM_MEMORY=2 TRT_NEED_FREE=0
TRT_EMOTION_VRAM="${BENCH_EMOTION_VRAM_MIB:-1536}"
[[ "$TRT_EMOTION_VRAM" =~ ^[0-9]+$ ]] || { echo 'BENCH_EMOTION_VRAM_MIB должен быть целым числом' >&2; exit 2; }
(( ! TRT_FULL )) || TRT_STREAM_MEMORY=5
if (( TRT_PARALLEL > 1 || TRT_FULL )); then
    TRT_MEMORY=$(( TRT_PARALLEL * TRT_STREAM_MEMORY + 2 ))
    (( TRT_MEMORY >= 8 )) || TRT_MEMORY=8
    export BENCH_CLIENT_MEMORY="${BENCH_CLIENT_MEMORY:-${TRT_MEMORY}g}"
fi
if (( TRT_PARALLEL > 1 )); then
    # Потоки делят ядра: по два на поток; сервер моделей работает на том же наборе.
    TRT_CORES=$(( TRT_PARALLEL * 2 ))
    export BENCH_CLIENT_CPUS="${BENCH_CLIENT_CPUS:-$TRT_CORES}" BENCH_CPUSET="${BENCH_CPUSET:-0-$(( TRT_CORES - 1 ))}"
fi
export BENCH_CLIENT_CPUS="${BENCH_CLIENT_CPUS:-$BENCH_THREADS}" BENCH_CPUSET="${BENCH_CPUSET:-0-3}"
export BENCH_CLIENT_MEMORY="${BENCH_CLIENT_MEMORY:-8g}"
export BENCH_TRITON_INSTANCES="${BENCH_TRITON_INSTANCES:-$(( (TRT_PARALLEL + 4) / 5 ))}"
export BENCH_GPU_MIN_FREE_MIB="${BENCH_GPU_MIN_FREE_MIB:-12288}" BENCH_GPU_MAX_UTIL="${BENCH_GPU_MAX_UTIL:-10}"
export BENCH_GPU_RESERVE_MIB="${BENCH_GPU_RESERVE_MIB:-3072}"
export BENCH_BUILD_NETWORK="${BENCH_BUILD_NETWORK:-default}" BENCH_DOWNLOAD_NETWORK="${BENCH_DOWNLOAD_NETWORK:-host}"
export BENCH_UID="$(id -u)" BENCH_GID="$(id -g)"
TRT_MIN_RAM="${BENCH_MIN_RAM_MIB:-24576}" TRT_MIN_DISK="${BENCH_MIN_DISK_MIB:-32768}"
TRT_WORKSPACE="${BENCH_TRT_WORKSPACE_MIB:-4096}" TRT_COMPILE_TIMEOUT="${BENCH_TRT_COMPILE_TIMEOUT:-3600}"
export BENCH_VAD_PROVIDER="${BENCH_VAD_PROVIDER:-cpu}" BENCH_SPEAKER_PROVIDER="${BENCH_SPEAKER_PROVIDER:-cpu}"
[[ "$BENCH_VAD_PROVIDER" == cpu || "$BENCH_VAD_PROVIDER" == cuda ]] || { echo 'BENCH_VAD_PROVIDER должен быть cpu или cuda' >&2; exit 2; }
[[ "$BENCH_SPEAKER_PROVIDER" == cpu || "$BENCH_SPEAKER_PROVIDER" == cuda ]] || { echo 'BENCH_SPEAKER_PROVIDER должен быть cpu или cuda' >&2; exit 2; }
for number in "$BENCH_GPU" "$BENCH_THREADS" "$BENCH_GPU_MIN_FREE_MIB" "$BENCH_GPU_MAX_UTIL" "$BENCH_GPU_RESERVE_MIB" "$TRT_MIN_RAM" "$TRT_MIN_DISK" "$TRT_WORKSPACE" "$TRT_COMPILE_TIMEOUT" "$BENCH_CLIENT_CPUS" "$BENCH_TRITON_INSTANCES"; do
    [[ "$number" =~ ^[0-9]+$ ]] || { echo 'Номер GPU, пороги и времена должны быть целыми числами' >&2; exit 2; }
done
(( BENCH_THREADS > 0 && BENCH_GPU_MAX_UTIL <= 100 && TRT_WORKSPACE >= 512 && TRT_WORKSPACE <= 8192 && TRT_COMPILE_TIMEOUT > 0 )) || exit 2
(( BENCH_CLIENT_CPUS >= 1 && BENCH_TRITON_INSTANCES >= 1 && BENCH_TRITON_INSTANCES <= 16 )) || { echo 'Ядер клиенту — не меньше одного, копий модели — от 1 до 16' >&2; exit 2; }
(( BENCH_CLIENT_CPUS <= $(nproc --all) )) || { echo "Запрошено ядер: $BENCH_CLIENT_CPUS, на машине $(nproc --all). Уменьшите --parallel или задайте BENCH_CLIENT_CPUS" >&2; exit 2; }
[[ "$BENCH_BUILD_NETWORK" == default || "$BENCH_BUILD_NETWORK" == host ]] || exit 2
[[ "$BENCH_DOWNLOAD_NETWORK" == default || "$BENCH_DOWNLOAD_NETWORK" == host ]] || exit 2
mkdir -p -- "$BENCH_OUT" "$BENCH_CACHE/gigaam-trt" "$BENCH_CACHE/gigaam"
export BENCH_OUT="$(cd -- "$BENCH_OUT" && pwd -P)" BENCH_CACHE="$(cd -- "$BENCH_CACHE" && pwd -P)"
case "$BENCH_OUT/" in "$BENCH_AUDIO_DIR/"*) echo 'Результаты должны быть вне папки аудио' >&2; exit 2 ;; esac
case "$BENCH_CACHE/" in "$BENCH_AUDIO_DIR/"*) echo 'Кеш должен быть вне папки аудио' >&2; exit 2 ;; esac
TRT_FULL_ARGS=() TRT_EXPORT_ARGS=() TRT_KIND=gigaam-trt
if (( TRT_FULL )); then
    TRT_FULL_ARGS=(--full) TRT_EXPORT_ARGS=(--emotions) TRT_KIND=gigaam-trt-full
    # Запас сверх резерва: сервер моделей и по одной модели эмоций на поток.
    TRT_NEED_FREE=$(( BENCH_GPU_RESERVE_MIB + 2048 + TRT_PARALLEL * TRT_EMOTION_VRAM ))
fi
TRT_RUN_ID="$TRT_KIND-$(TZ=Europe/Moscow date +%Y%m%d-%H%M%S)-$$"
export BENCH_RUN_OUT="$BENCH_OUT/$TRT_RUN_ID"
export BENCH_REFERENCE_DIR="${TRT_REFERENCE:-$TRT_PROJECT_DIR/benchmark}"
mkdir -p "$BENCH_RUN_OUT/логи"
TRT_OWN_FILE="$BENCH_RUN_OUT/логи/контейнеры.txt"
: > "$TRT_OWN_FILE"
exec 3>&1 4>&2
exec >>"$BENCH_RUN_OUT/логи/запуск.log" 2>&1
tail --pid=$$ --follow=descriptor --sleep-interval=0.2 --lines=+1 "$BENCH_RUN_OUT/логи/запуск.log" >&3 2>&4 9>&- &
TRT_CONSOLE_PID=$! TRT_MONITOR_PID='' TRT_LOG_PID=''
trt_compose() { docker compose --project-name speech-comparison-gigaam-trt -f "$TRT_PROJECT_DIR/compose.gigaam-trt-benchmark.yml" "$@"; }
stop_owned() {
    local container project
    while IFS= read -r container; do
        [[ "$container" =~ ^[a-f0-9]{12,64}$ ]] || continue
        project=$(timeout 10 docker inspect --format '{{index .Config.Labels "com.docker.compose.project"}}' "$container" 2>/dev/null) || continue
        [[ "$project" == speech-comparison-gigaam-trt ]] || continue
        timeout 30 docker stop --time 5 "$container" >/dev/null 2>&1 || true
        timeout 15 docker inspect "$container" > "$BENCH_RUN_OUT/логи/контейнер-$container.json" 2>&1 || true
        timeout 30 docker logs "$container" > "$BENCH_RUN_OUT/логи/контейнер-$container.log" 2>&1 || true
    done < "$TRT_OWN_FILE"
}
stop_log_reader() {
    [[ -n "$TRT_LOG_PID" ]] || return 0
    kill "$TRT_LOG_PID" 2>/dev/null || true
    # Не ждём бесконечно docker logs после Exited: он ранее удерживал прогон.
    for _ in 1 2 3 4; do
        kill -0 "$TRT_LOG_PID" 2>/dev/null || break
        sleep .2
    done
    kill -KILL "$TRT_LOG_PID" 2>/dev/null || true
    wait "$TRT_LOG_PID" 2>/dev/null || true
    TRT_LOG_PID=''
}
cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM HUP
    [[ -z "$TRT_MONITOR_PID" ]] || kill "$TRT_MONITOR_PID" 2>/dev/null || true
    stop_log_reader
    stop_owned
    echo "Завершение: код $exit_code. Контейнеры нашего прогона остановлены, автоматическое удаление отключено."
    # Финализация работает штатным Python сервера, без pip и без моделей.
    (cd -- "$TRT_PROJECT_DIR" && PYTHONDONTWRITEBYTECODE=1 python3 -m benchmark.gigaam_trt_run \
        --finalize "$BENCH_RUN_OUT" "$exit_code") || {
        echo 'Архив не создан; подробные журналы сохранены в папке прогона.'
        (( exit_code != 0 )) || exit_code=1
    }
    echo "Результаты: $BENCH_RUN_OUT"
    sleep .3
    kill "$TRT_CONSOLE_PID" 2>/dev/null || true
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
    (( free >= BENCH_GPU_MIN_FREE_MIB && util <= BENCH_GPU_MAX_UTIL && ram >= TRT_MIN_RAM )) || return 1
    if (( free < TRT_NEED_FREE )); then
        echo "Полному flow на $TRT_PARALLEL потоков нужно $TRT_NEED_FREE МиБ свободной видеопамяти: у каждого потока своя модель эмоций. Уменьшите --parallel."
        return 1
    fi
    docker_root=$(timeout 30 docker info --format '{{.DockerRootDir}}')
    for directory in "$BENCH_OUT" "$BENCH_CACHE" "$docker_root"; do
        disk=$(df -Pm "$directory" | awk 'NR==2 {print $4}')
        echo "Диск $directory: свободно $disk МиБ."
        [[ "$disk" =~ ^[0-9]+$ ]] && (( disk >= TRT_MIN_DISK )) || return 1
    done
}
monitor_gpu() {
    echo 'timestamp,uuid,utilization_percent,used_mib,free_mib,total_mib,power_w,temperature_c' > "$BENCH_RUN_OUT/логи/gpu.csv"
    while true; do
        local sample free
        sample=$(timeout 10 nvidia-smi -i "$BENCH_GPU" --query-gpu=timestamp,uuid,utilization.gpu,memory.used,memory.free,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits) || sample=''
        echo "$sample" >> "$BENCH_RUN_OUT/логи/gpu.csv"
        free=$(awk -F, '{gsub(/ /,"",$5); print $5}' <<< "$sample")
        if [[ ! "$free" =~ ^[0-9]+$ ]] || (( free < BENCH_GPU_RESERVE_MIB )); then
            echo "Нарушен резерв GPU или сведения недоступны: $free МиБ; минимум $BENCH_GPU_RESERVE_MIB. Останавливаем только наш прогон." \
                | tee "$BENCH_RUN_OUT/логи/остановка-по-памяти.txt"
            stop_owned
            return
        fi
        sleep 2
    done
}
start_container() {
    local name=$1 service=$2 id
    shift 2
    [[ ! -f "$BENCH_RUN_OUT/логи/остановка-по-памяти.txt" ]] || return 42
    trt_compose run --detach --no-deps --pull never --use-aliases --name "$name" "$service" "$@"
    id=$(timeout 10 docker inspect --format '{{.Id}}' "$name")
    [[ "$id" =~ ^[a-f0-9]{12,64}$ ]] || return 1
    printf '%s\n' "$id" >> "$TRT_OWN_FILE"
}
wait_task() {
    local name=$1 title=$2 state running exit_code started=$SECONDS
    docker logs --follow "$name" 9>&- &
    TRT_LOG_PID=$!
    while true; do
        [[ ! -f "$BENCH_RUN_OUT/логи/остановка-по-памяти.txt" ]] || return 42
        state=$(timeout 10 docker inspect --format '{{.State.Running}} {{.State.ExitCode}}' "$name")
        read -r running exit_code <<< "$state"
        if [[ "$running" == false ]]; then
            stop_log_reader
            timeout 30 docker logs "$name" > "$BENCH_RUN_OUT/логи/$title.log" 2>&1 || true
            echo "Задача $name завершена: код $exit_code."
            [[ ! -f "$BENCH_RUN_OUT/логи/остановка-по-памяти.txt" ]] || return 42
            return "$exit_code"
        fi
        if (( SECONDS - started >= 86400 )); then
            echo "Задача $name превысила 24 часа; останавливаем свой прогон."
            return 124
        fi
        sleep 2
    done
}
if (( TRT_FULL )); then
    echo "Прогон $TRT_RUN_ID: GigaAM TensorRT — поиск речи, распознавание, голоса собеседников и эмоции; LLM не запускается."
    echo "Голоса: CAM++ на $BENCH_SPEAKER_PROVIDER; эмоции: GigaAM-Emo на GPU, своя модель у каждого потока, нужно свободной видеопамяти: $TRT_NEED_FREE МиБ."
else
    echo "Прогон $TRT_RUN_ID: GigaAM TensorRT — поиск речи, распознавание и паузы; голоса, эмоции и LLM не запускаются."
fi
echo "GPU $BENCH_GPU; поиск речи: $BENCH_VAD_PROVIDER; модели в $BENCH_CACHE/gigaam-trt; результаты $BENCH_RUN_OUT."
echo "Одновременно записей: $TRT_PARALLEL; копий модели в Triton: $BENCH_TRITON_INSTANCES; ядер: $BENCH_CLIENT_CPUS (набор $BENCH_CPUSET); память клиента: $BENCH_CLIENT_MEMORY."
echo 'Экспорт модели и сборка TensorRT — вне замеров. Обработка: внутренняя сеть, без публикации портов.'
active=$(timeout 15 docker ps -q --filter label=com.docker.compose.project=speech-comparison-gigaam-trt)
if [[ -n "$active" ]]; then
    echo 'Остался работающий контейнер прежнего теста GigaAM TensorRT; новый запуск не начинаем.'
    docker ps --filter label=com.docker.compose.project=speech-comparison-gigaam-trt --format 'table {{.Names}}\t{{.Status}}'
    exit 2
fi
if ! check_capacity; then echo 'Недостаточно ресурсов; модели не запускались.'; exit 42; fi
# docker load позволяет перенести тот же официальный образ с другой машины.
# Если он уже есть, не обращаемся к nvcr.io даже за проверкой манифеста.
if docker image inspect "$TRT_IMAGE" >/dev/null 2>&1; then
    echo "Официальный образ Triton уже есть локально: $TRT_IMAGE. Повторного скачивания не будет."
else
    echo "Загружаем официальный образ Triton: $TRT_IMAGE — до сборки клиента и запуска моделей."
    if ! docker pull "$TRT_IMAGE"; then
        echo "Не удалось скачать $TRT_IMAGE. Обработка записей не начиналась."
        echo 'Если реестр отвечает 403 Forbidden, доступ отклонён; код ответа не устанавливает точную причину.'
        echo 'Нужен доступ к nvcr.io либо перенос этого официального образа через docker image save / docker image load.'
        echo 'После переноса повторите ту же команду. Удалять прошлые результаты и кеш не требуется.'
        echo 'Образ можно загрузить через прокси: python3 benchmark/fetch_triton_image.py --load (см. benchmark/TONE-TRT.md).'
        exit 1
    fi
fi
if ! docker image inspect speech-comparison:4.0.0-gigaam-cuda >/dev/null 2>&1; then
    echo 'Собираем общий образ библиотек TensorRT-клиентов; сервисы не запускаются.'
    docker compose --project-name speech-comparison-gigaam-trt -f "$TRT_PROJECT_DIR/compose.trt-base.yml" \
        --progress plain build base
fi
trt_compose --progress plain build client
docker image inspect speech-comparison:2.0.0-gigaam-trt-client "$TRT_IMAGE" \
    > "$BENCH_RUN_OUT/логи/образы.json"
if ! check_capacity; then echo 'После сборки недостаточно ресурсов.'; exit 42; fi
start_container "speech-comparison-gigaam-trt-export-$TRT_RUN_ID" export \
    export --out /results "${TRT_EXPORT_ARGS[@]}"
wait_task "speech-comparison-gigaam-trt-export-$TRT_RUN_ID" экспорт
monitor_gpu 9>&- &
TRT_MONITOR_PID=$!
start_container "speech-comparison-gigaam-trt-compile-$TRT_RUN_ID" compile \
    compile --out /results --workspace-mib "$TRT_WORKSPACE" --timeout "$TRT_COMPILE_TIMEOUT" --instances "$BENCH_TRITON_INSTANCES"
wait_task "speech-comparison-gigaam-trt-compile-$TRT_RUN_ID" сборка-tensorrt
TRT_SERVER="speech-comparison-gigaam-trt-server-$TRT_RUN_ID"
start_container "$TRT_SERVER" triton
ready_started=$SECONDS
while true; do
    state=$(timeout 10 docker inspect --format '{{.State.Running}} {{.State.Health.Status}}' "$TRT_SERVER")
    read -r running health <<< "$state"
    if [[ "$running" != true || "$health" == unhealthy ]] || (( SECONDS - ready_started > 600 )); then
        timeout 30 docker logs "$TRT_SERVER" > "$BENCH_RUN_OUT/логи/triton.log" 2>&1 || true
        cat "$BENCH_RUN_OUT/логи/triton.log"
        echo 'Наш Triton не готов; замеры не начинались.'
        exit 1
    fi
    [[ ! -f "$BENCH_RUN_OUT/логи/остановка-по-памяти.txt" ]] || exit 42
    [[ "$health" == healthy ]] && break
    echo "Ждём готовности нашего Triton: $health — вне замеров."
    sleep 5
done
echo 'Наш Triton готов; начинаем корпус. Загрузка серверной модели исключена из замеров.'
start_container "speech-comparison-gigaam-trt-client-$TRT_RUN_ID" client /recordings --parallel "$TRT_PARALLEL" \
    --out /results "${TRT_FULL_ARGS[@]}" "${TRT_REFERENCE_ARGS[@]}" "${TRT_ARGS[@]}"
wait_task "speech-comparison-gigaam-trt-client-$TRT_RUN_ID" gigaam-trt
timeout 30 docker logs "$TRT_SERVER" > "$BENCH_RUN_OUT/логи/triton.log" 2>&1 || true
