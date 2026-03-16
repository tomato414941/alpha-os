#!/bin/bash
set -euo pipefail
ASSETS="${1:-BTC}"
cd /home/dev/projects/alpha-os
source .venv/bin/activate
source ~/.secrets/binance
exec python3 -m alpha_os trade --schedule --assets "$ASSETS" --config /home/dev/.config/alpha-os/prod.toml 2>&1
