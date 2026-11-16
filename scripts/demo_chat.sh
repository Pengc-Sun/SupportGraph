#!/usr/bin/env bash
# 验收:两轮流式对话,第二轮须续上第一轮上下文(ch02 新契约:user_id + conversation_id)
set -euo pipefail
UID_DEMO="demo-$$"
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

echo "=== 第一轮:自报家门+提商品 ==="
curl --fail -sN http://localhost:8000/api/chat -H 'Content-Type: application/json' \
  -d "{\"user_id\": \"$UID_DEMO\", \"message\": \"我叫王小明,昨天买了你们的智能猫砂盆\"}" | tee "$TMP"
CID=$(grep -o '"conversation_id": [0-9]*' "$TMP" | tail -1 | grep -o '[0-9]*')
if [ -z "$CID" ]; then
  echo "错误:未能从第一轮 SSE 输出中提取 conversation_id" >&2
  exit 1
fi
echo; echo "=== 第二轮:考上下文(我叫什么?买了什么?)conversation_id=$CID ==="
curl --fail -sN http://localhost:8000/api/chat -H 'Content-Type: application/json' \
  -d "{\"user_id\": \"$UID_DEMO\", \"message\": \"还记得我叫什么、买了什么吗?\", \"conversation_id\": $CID}"
echo
