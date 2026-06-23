#!/usr/bin/env bash
#
# Развёртывание ШАУ из GitHub на ЧИСТЫЙ сервер (Ubuntu/Debian).
# Ставит зависимости, клонирует репозиторий и поднимает сервис. Идемпотентно:
# повторный запуск подтягивает свежий код и перезапускает сервис.
#
# Быстрый старт (на сервере, под root или с sudo):
#   curl -fsSL https://raw.githubusercontent.com/mnazarow/shau-norming/main/deploy_from_github.sh | sudo bash
# либо:
#   sudo ./deploy_from_github.sh
#
# Параметры (env или флаги):
#   REPO       URL репозитория            (--repo)   [https://github.com/mnazarow/shau-norming.git]
#   BRANCH     ветка                      (--branch) [main]
#   DIR        каталог установки          (--dir)    [/opt/shau-norming]
#   HOST_PORT  внешний порт               (--port)   [8000]
#   MODE       docker | native | auto     (--native/--docker) [auto: docker если возможно]
#
set -euo pipefail

REPO="${REPO:-https://github.com/mnazarow/shau-norming.git}"
BRANCH="${BRANCH:-main}"
DIR="${DIR:-/opt/shau-norming}"
HOST_PORT="${HOST_PORT:-8000}"
MODE="${MODE:-auto}"

# ---- разбор флагов ----
while [ $# -gt 0 ]; do
  case "$1" in
    --repo) REPO="$2"; shift 2;;
    --branch) BRANCH="$2"; shift 2;;
    --dir) DIR="$2"; shift 2;;
    --port) HOST_PORT="$2"; shift 2;;
    --docker) MODE="docker"; shift;;
    --native) MODE="native"; shift;;
    http*://*|git@*) REPO="$1"; shift;;
    *) echo "Неизвестный аргумент: $1"; exit 1;;
  esac
done

# ---- нужен root ----
if [ "$(id -u)" -ne 0 ]; then
  echo ">> Требуются права root — перезапуск через sudo…"
  exec sudo -E REPO="$REPO" BRANCH="$BRANCH" DIR="$DIR" HOST_PORT="$HOST_PORT" MODE="$MODE" bash "$0"
fi

export DEBIAN_FRONTEND=noninteractive

echo ">> [1/4] Базовые пакеты (git, curl)…"
apt-get update -y
apt-get install -y git ca-certificates curl

echo ">> [2/4] Получение кода: $REPO ($BRANCH) -> $DIR"
if [ -d "$DIR/.git" ]; then
  git -C "$DIR" remote set-url origin "$REPO"
  git -C "$DIR" fetch --depth 1 origin "$BRANCH"
  git -C "$DIR" checkout -B "$BRANCH" "origin/$BRANCH"
else
  rm -rf "$DIR"
  git clone --depth 1 -b "$BRANCH" "$REPO" "$DIR"
fi
cd "$DIR"

# ---- выбор режима ----
if [ "$MODE" = "auto" ]; then
  MODE="docker"
fi

if [ "$MODE" = "docker" ]; then
  echo ">> [3/4] Docker…"
  if ! command -v docker >/dev/null 2>&1; then
    echo "   установка Docker (get.docker.com)…"
    curl -fsSL https://get.docker.com | sh
  fi
  if ! docker compose version >/dev/null 2>&1; then
    apt-get install -y docker-compose-plugin || true
  fi
  systemctl enable --now docker 2>/dev/null || true
  echo ">> [4/4] Сборка и запуск контейнера (порт $HOST_PORT)…"
  HOST_PORT="$HOST_PORT" docker compose up -d --build
  echo ">> Контейнер запущен (restart=unless-stopped — переживёт перезагрузку)."
else
  echo ">> [3/4] Python + venv…"
  apt-get install -y python3 python3-venv python3-pip
  echo ">> [4/4] Установка зависимостей и systemd-сервиса (порт $HOST_PORT)…"
  chmod +x run_server.sh
  PORT="$HOST_PORT" HOST=0.0.0.0 ./run_server.sh --systemd
fi

IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo ""
echo "================================================================"
echo " ШАУ развёрнут. Откройте:  http://${IP:-<server-ip>}:$HOST_PORT"
echo " Обновить:   sudo $DIR/deploy_from_github.sh"
if [ "$MODE" = "docker" ]; then
  echo " Логи:       docker compose -f $DIR/docker-compose.yml logs -f"
  echo " Остановить: docker compose -f $DIR/docker-compose.yml down"
else
  echo " Статус:     systemctl status shau-norming"
  echo " Логи:       journalctl -u shau-norming -f"
fi
echo "================================================================"
