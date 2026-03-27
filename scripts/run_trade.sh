#!/bin/bash
set -euo pipefail
ASSET="${1:-BTC}"
cd /home/dev/projects/alpha-os
source .venv/bin/activate
source ~/.secrets/binance
python3 -m alpha_os_recovery trade --once --asset "$ASSET" --config /home/dev/.config/alpha-os/prod.toml 2>&1
