#!/usr/bin/env bash
# Gom code dashboard + file deploy vào thư mục tạm rồi upload lên HF Space.
# Dùng (từ gốc repo, đã `hf auth login`):
#   bash deploy/hf-space/push.sh                 # mặc định kkoi31/airline-dw-dashboard
#   SPACE=user/ten-space bash deploy/hf-space/push.sh
#   STAGE_ONLY=1 bash deploy/hf-space/push.sh    # chỉ gom file (để docker build thử), không upload
set -euo pipefail

SPACE="${SPACE:-kkoi31/airline-dw-dashboard}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HF="${HF:-$ROOT/venv/bin/hf}"
STAGE="${STAGE:-$(mktemp -d)}"

cp "$ROOT"/deploy/hf-space/{Dockerfile,README.md,requirements.txt,fetch_data.py} "$STAGE"/
mkdir -p "$STAGE/dashboard" "$STAGE/mining"
cp "$ROOT"/dashboard/*.py "$STAGE/dashboard/"
cp "$ROOT/mining/predict_delay.py" "$STAGE/mining/"

echo "Staged → $STAGE"
[ "${STAGE_ONLY:-0}" = "1" ] && exit 0

"$HF" repos create "$SPACE" --type space --sdk docker --exist-ok
"$HF" upload "$SPACE" "$STAGE" . --repo-type space --commit-message "Deploy dashboard"
echo "Space: https://huggingface.co/spaces/$SPACE"
