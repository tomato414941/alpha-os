#!/bin/bash
set -euo pipefail

# Map portfolio group names to asset lists
RAW="${1:-crypto-spot}"
case "$RAW" in
  crypto-spot) ASSETS="BTC,ETH,SOL" ;;
  *)           ASSETS="$RAW" ;;
esac

cd /home/dev/projects/alpha-os
source .venv/bin/activate
source ~/.secrets/binance
exec python3 -m alpha_os trade --schedule --assets "$ASSETS" --config /home/dev/.config/alpha-os/prod.toml 2>&1
