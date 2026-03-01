#!/bin/bash
set -euo pipefail
ASSET="${1:-BTC}"
cd /home/dev/projects/alpha-os
source .venv/bin/activate
source ~/.secrets/binance
exec python3 -m alpha_os live --schedule --asset "$ASSET" 2>&1
