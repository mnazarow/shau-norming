#!/usr/bin/env bash
#
# Запуск веб-интерфейса нормирования ШАУ на чистом сервере (Ubuntu/Debian) БЕЗ Docker.
# Создаёт виртуальное окружение, ставит зависимости и поднимает сервер.
#
# Использование:
#   ./run_server.sh                 # порт 8000, foreground
#   PORT=8080 ./run_server.sh        # другой порт
#   ./run_server.sh --systemd        # установить как сервис systemd (нужен root)
#
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
VENV=".venv"

# 1) Системные пакеты (если есть apt и прав достаточно)
if command -v apt-get >/dev/null 2>&1; then
  if ! command -v python3 >/dev/null 2>&1 || ! python3 -m venv --help >/dev/null 2>&1; then
    echo ">> Устанавливаю python3/venv (нужен sudo)…"
    sudo apt-get update -y
    sudo apt-get install -y python3 python3-venv python3-pip
  fi
fi

# 2) Виртуальное окружение и зависимости
if [ ! -d "$VENV" ]; then
  echo ">> Создаю venv…"
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
echo ">> Устанавливаю зависимости…"
pip install --upgrade pip -q
pip install -q -r requirements.txt
# CatBoost ставится отдельно (необязательно): pip install catboost

# 3) Установка как systemd-сервис (опционально)
if [ "${1:-}" = "--systemd" ]; then
  UNIT=/etc/systemd/system/shau-norming.service
  echo ">> Устанавливаю systemd-сервис $UNIT (нужен sudo)…"
  sudo tee "$UNIT" >/dev/null <<EOF
[Unit]
Description=ШАУ — веб-интерфейс нормирования трудоёмкости
After=network.target

[Service]
Type=simple
WorkingDirectory=$(pwd)
ExecStart=$(pwd)/$VENV/bin/python app.py --host $HOST --port $PORT
Restart=always
User=$(whoami)

[Install]
WantedBy=multi-user.target
EOF
  sudo systemctl daemon-reload
  sudo systemctl enable --now shau-norming
  echo ">> Сервис запущен. Статус:  sudo systemctl status shau-norming"
  exit 0
fi

# 4) Запуск в текущем терминале
echo ">> Старт:  http://$HOST:$PORT   (Ctrl+C — остановить)"
exec python app.py --host "$HOST" --port "$PORT"
