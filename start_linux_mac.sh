#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -d .venv ]]; then python3 -m venv .venv; fi
source .venv/bin/activate
python -m pip install -e .
echo 'Abra http://127.0.0.1:8000 no navegador. Para parar, use Ctrl+C.'
uvicorn carsel.api:app --host 127.0.0.1 --port 8000
