#!/bin/bash
set -euo pipefail
ASSETS="${1:-BTC}"
cd /home/dev/projects/alpha-os
source .venv/bin/activate
source ~/.secrets/binance
exec python3 -m alpha_os live --schedule --assets "$ASSETS" 2>&1
