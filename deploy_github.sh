#!/usr/bin/env bash
#
# Публикация проекта на GitHub под ВАШИМ аккаунтом.
#
# Вариант A (рекомендуется) — через GitHub CLI (gh):
#     gh auth login            # один раз
#     ./deploy_github.sh shau-norming            # создаст приватный репо и запушит
#     ./deploy_github.sh shau-norming --public   # публичный
#
# Вариант B — если репозиторий на GitHub уже создан вручную:
#     ./deploy_github.sh https://github.com/USER/REPO.git
#
set -euo pipefail
cd "$(dirname "$0")"

ARG="${1:?Укажите имя репозитория или URL. Пример: ./deploy_github.sh shau-norming}"
VIS="private"; [ "${2:-}" = "--public" ] && VIS="public"

# git-история (если ещё не инициализирована)
if [ ! -d .git ]; then
  git init -b main
fi
git add -A
git commit -m "ШАУ: нормирование трудоёмкости — веб-интерфейс, парсер, ML, деплой" || true

if [[ "$ARG" == http*://* || "$ARG" == git@* ]]; then
  # Вариант B: готовый URL
  git remote remove origin 2>/dev/null || true
  git remote add origin "$ARG"
  git push -u origin main
else
  # Вариант A: создаём репозиторий через gh
  if ! command -v gh >/dev/null 2>&1; then
    echo "GitHub CLI (gh) не установлен. Установите: https://cli.github.com/"
    echo "или создайте репозиторий вручную и запустите:"
    echo "  ./deploy_github.sh https://github.com/USER/$ARG.git"
    exit 1
  fi
  gh repo create "$ARG" --"$VIS" --source=. --remote=origin --push
fi

echo ">> Готово. Репозиторий опубликован."
