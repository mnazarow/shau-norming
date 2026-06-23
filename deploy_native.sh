#!/usr/bin/env bash
#
# Развёртывание ШАУ из GitHub на ЧИСТЫЙ сервер БЕЗ Docker.
# Самодостаточный скрипт: apt-зависимости -> клон репозитория -> venv ->
# systemd-сервис под отдельным пользователем. Идемпотентный: повторный запуск
# обновляет код и зависимости и перезапускает сервис.
#
# Быстрый старт (Ubuntu/Debian, под root/sudo):
#   curl -fsSL https://raw.githubusercontent.com/mnazarow/shau-norming/main/deploy_native.sh | sudo bash
# либо:
#   sudo ./deploy_native.sh
#
# Параметры (env или флаги):
#   REPO      URL репозитория        (--repo)    [https://github.com/mnazarow/shau-norming.git]
#   BRANCH    ветка                  (--branch)  [main]
#   DIR       каталог установки      (--dir)     [/opt/shau-norming]
#   PORT      порт сервиса           (--port)    [8000]
#   SERVICE   имя systemd-сервиса    (--service) [shau-norming]
#   RUN_USER  пользователь сервиса   (--user)    [shau]
#   WITH_ML=1 ставить CatBoost       (--with-ml) [нет: используется numpy-fallback]
#
set -euo pipefail

REPO="${REPO:-https://github.com/mnazarow/shau-norming.git}"
BRANCH="${BRANCH:-main}"
DIR="${DIR:-/opt/shau-norming}"
PORT="${PORT:-8000}"
SERVICE="${SERVICE:-shau-norming}"
RUN_USER="${RUN_USER:-shau}"
WITH_ML="${WITH_ML:-0}"

while [ $# -gt 0 ]; do
  case "$1" in
    --repo) REPO="$2"; shift 2;;
    --branch) BRANCH="$2"; shift 2;;
    --dir) DIR="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --service) SERVICE="$2"; shift 2;;
    --user) RUN_USER="$2"; shift 2;;
    --with-ml) WITH_ML=1; shift;;
    http*://*|git@*) REPO="$1"; shift;;
    *) echo "Неизвестный аргумент: $1"; exit 1;;
  esac
done

# ---- нужен root ----
if [ "$(id -u)" -ne 0 ]; then
  echo ">> Требуются права root — перезапуск через sudo…"
  exec sudo -E REPO="$REPO" BRANCH="$BRANCH" DIR="$DIR" PORT="$PORT" \
       SERVICE="$SERVICE" RUN_USER="$RUN_USER" WITH_ML="$WITH_ML" bash "$0"
fi

export DEBIAN_FRONTEND=noninteractive

echo ">> [1/6] Системные пакеты (git, python3, venv, pip)…"
apt-get update -y
apt-get install -y git ca-certificates curl python3 python3-venv python3-pip

echo ">> [2/6] Получение кода: $REPO ($BRANCH) -> $DIR"
if [ -d "$DIR/.git" ]; then
  git -C "$DIR" remote set-url origin "$REPO"
  git -C "$DIR" fetch --depth 1 origin "$BRANCH"
  git -C "$DIR" checkout -B "$BRANCH" "origin/$BRANCH"
else
  rm -rf "$DIR"
  git clone --depth 1 -b "$BRANCH" "$REPO" "$DIR"
fi
cd "$DIR"

echo ">> [3/6] Пользователь сервиса: $RUN_USER"
if ! id -u "$RUN_USER" >/dev/null 2>&1; then
  useradd --system --no-create-home --shell /usr/sbin/nologin "$RUN_USER"
fi

echo ">> [4/6] Виртуальное окружение и зависимости…"
python3 -m venv "$DIR/.venv"
"$DIR/.venv/bin/pip" install --upgrade pip -q
"$DIR/.venv/bin/pip" install -q -r "$DIR/requirements.txt"
if [ "$WITH_ML" = "1" ]; then
  echo "   установка CatBoost (полный ML)…"
  "$DIR/.venv/bin/pip" install -q catboost || echo "   [!] CatBoost не установился — будет numpy-fallback"
fi
chown -R "$RUN_USER":"$RUN_USER" "$DIR"

echo ">> [5/6] systemd-сервис /etc/systemd/system/$SERVICE.service…"
tee "/etc/systemd/system/$SERVICE.service" >/dev/null <<EOF
[Unit]
Description=ШАУ — веб-интерфейс нормирования трудоёмкости (native)
After=network.target

[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=$DIR
ExecStart=$DIR/.venv/bin/python app.py --host 0.0.0.0 --port $PORT
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

echo ">> [6/6] Запуск сервиса…"
systemctl daemon-reload
systemctl enable "$SERVICE" >/dev/null 2>&1 || true
systemctl restart "$SERVICE"
sleep 1
systemctl --no-pager --lines=0 status "$SERVICE" || true

IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo ""
echo "================================================================"
echo " ШАУ развёрнут (native, без Docker)."
echo " Откройте:    http://${IP:-<server-ip>}:$PORT"
echo " Статус:      systemctl status $SERVICE"
echo " Логи:        journalctl -u $SERVICE -f"
echo " Обновить:    sudo $DIR/deploy_native.sh"
echo " Остановить:  systemctl stop $SERVICE"
echo "================================================================"
