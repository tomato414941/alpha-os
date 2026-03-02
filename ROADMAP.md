# Multi-Timeframe Alpha System Roadmap

alpha-os + signal-noise の統合進化ロードマップ。
現在の日次シグナル × 4h サイクルから、マルチタイムフレーム・イベント駆動型への移行計画。

## Vision

市場の情報は「半減期」を持つ。日次でしか動かない情報もあれば、
数分で織り込まれる情報もある。理想的なシステムは情報の鮮度に応じた
複数のレイヤーで動作する：

```
Layer 3: Strategic（日次）  — 方向性バイアスを決定
Layer 2: Tactical（時間）   — エントリー/エグジットの窓を特定
Layer 1: Execution（分）    — 最適な瞬間に最小コストで執行
```

各レイヤーが独立した GP 進化 + ライフサイクルを持ち、
上位レイヤーの判断を下位が実行する階層構造。

## Current State (2026-03-02)

### alpha-os

- 3-Layer architecture: Strategic (日次) / Tactical (時間) / Execution (分)
- S-expression DSL + GP 進化 + MAP-Elites アーカイブ
- Layer 2: TacticalTrader（hourly signals, funding rate, OI, liquidations）
- Layer 1: ExecutionOptimizer（VPIN, spread, imbalance ベース執行最適化）
- EventDrivenTrader（WebSocket イベント駆動 + デバウンス）
- Distributional risk layer（CVaR/left-tail gate + fractional Kelly sizing）
- BinanceExecutor（spot, testnet）+ optimizer 連携
- シグナル源: signal-noise REST API + WebSocket（日次 + hourly + 1min realtime）

### signal-noise

- 1,256+ collector、10 domain、asyncio スケジューラ
- SQLite (WAL mode) + FastAPI REST API + WebSocket (`/ws/signals`)
- EventBus（in-process pub/sub）
- StreamingCollector（WebSocket ベース）: orderbook, trade flow, liquidation, funding rate
- `signals_realtime` テーブル（1分足、30日保持 + 日次ロールアップ）
- Realtime API（`/signals/{name}/realtime`）

### Phase Status

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | **完了** | Tactical Data Layer（hourly collectors, subdaily DataStore, Layer 2 GP）|
| Phase 2 | **完了** | Event-Driven Architecture（EventBus, WebSocket, StreamingCollector）|
| Phase 3 | **完了** | Microstructure Layer（orderbook/VPIN, ExecutionOptimizer, DSL templates）|
| Phase 4 | 未着手 | Options Intelligence（Deribit IV/skew, IV-aware risk scaling）|
| Phase 5 | 未着手 | Cross-Exchange Intelligence（multi-exchange, lead-lag alpha）|

---

## Phase 1: Tactical Data Layer — 時間足シグナル ✅ 完了

**目標**: 1h〜8h 周期の情報源を追加し、4h サイクルに実質的な意味を持たせる。

### 1.1 signal-noise: 新規 Collector 追加

#### Funding Rate（優先度: 最高）

```
collector: binance_funding_rate
domain: financial
category: crypto_derivatives
frequency: hourly (8h で確定、予測値は常時更新)
source: Binance Futures API (GET /fapi/v1/fundingRate)
data: [timestamp, symbol, fundingRate, fundingCountdown]
signals: funding_rate_btc, funding_rate_eth, funding_rate_sol
```

- Binance API は public（認証不要、L1）
- 確定値 + 次回予測値の両方を保存
- funding rate の**変化速度**（velocity）を computed signal として派生

#### Liquidation Data（優先度: 最高）

```
collector: binance_liquidations
domain: financial
category: crypto_derivatives
frequency: hourly (1h 集計)
source: Binance Futures API (GET /fapi/v1/allForceOrders)
data: [timestamp, symbol, side, qty, price, total_value]
signals: liq_long_btc_1h, liq_short_btc_1h, liq_ratio_btc_1h
```

- Public API（L1）、1h ごとにロング/ショート清算額を集計
- liq_ratio = long_liq / (long_liq + short_liq) で方向性を示す

#### Open Interest（優先度: 高）

```
collector: binance_open_interest
domain: financial
category: crypto_derivatives
frequency: hourly
source: Binance Futures API (GET /fapi/v1/openInterest)
data: [timestamp, symbol, openInterest, notionalValue]
signals: oi_btc, oi_eth, oi_sol, oi_total
```

- OI の急増/急減はレバレッジの蓄積/解消シグナル

#### Long/Short Ratio（優先度: 高）

```
collector: binance_long_short_ratio
domain: financial
category: crypto_derivatives
frequency: hourly
source: Binance Futures API
  - /futures/data/globalLongShortAccountRatio
  - /futures/data/topLongShortAccountRatio
  - /futures/data/topLongShortPositionRatio
data: [timestamp, symbol, longAccount, shortAccount, longShortRatio]
signals: ls_ratio_global_btc, ls_ratio_top_btc, ls_position_ratio_btc
```

#### CEX Netflow（優先度: 中）

```
collector: coinglass_exchange_netflow
domain: financial
category: crypto_flow
frequency: hourly
source: Coinglass API (requires key, L2)
data: [timestamp, exchange, symbol, inflow, outflow, netflow]
signals: netflow_btc_binance, netflow_btc_total
```

- 取引所への入金 = 売り圧力の先行指標
- 代替: CryptoQuant (L2)、Glassnode (L2)

#### Stablecoin Supply（優先度: 中）

```
collector: stablecoin_supply
domain: financial
category: crypto_flow
frequency: hourly
source: DefiLlama API (public, L1)
data: [timestamp, token, circulating, minted_24h, burned_24h]
signals: usdt_supply, usdc_supply, stablecoin_mint_rate
```

- mint 急増 = 「買い弾」の準備

### 1.2 signal-noise: スケジューラ拡張

現状の asyncio スケジューラは hourly が最小粒度で、実装上はサブ時間もサポート可能。
変更点：

- `collect_interval` を 300s（5分）まで対応
  - 現状も秒単位で指定可能だが、テスト・運用実績がない
- Rate limit 管理の強化
  - Binance Futures API: 2400 req/min（十分余裕あり）
  - 複数 collector が同一 API を叩く場合の provider 単位レートリミッタ
- `signal_meta.interval` の更新
  - hourly collector は `interval=3600` で登録

### 1.3 signal-noise: API 拡張

```
GET /signals/{name}/data?since=2026-03-01T00:00:00&resolution=1h
```

- `resolution` パラメータ追加: `1m`, `5m`, `1h`, `4h`, `1d`
- 高頻度シグナルは `resolution` でリサンプリング（OHLCV は OHLCV 集約、scalar は last）
- alpha-os の DataStore.sync() がサブデイリーデータを取得できるように

### 1.4 alpha-os: DataStore のサブデイリー対応

```python
# Before (daily only)
store.get_matrix(features, end="2026-03-01")

# After (resolution-aware)
store.get_matrix(features, end="2026-03-01T12:00:00", resolution="1h")
```

- `alpha_cache.db` に `resolution` カラム追加（`1d` がデフォルト）
- `get_matrix()` が resolution に応じて ffill/resample
- 既存の日次パスは完全に後方互換

### 1.5 alpha-os: DSL のタイムスケール対応

現状の window は「日数」を意味する（`mean_20` = 20日移動平均）。
サブデイリーでは window の意味が変わる：

```
# Strategy: 解釈は Layer 依存
# Layer 3 DSL: (mean_20 btc_ohlcv) → 20日平均
# Layer 2 DSL: (mean_20 funding_rate_btc) → 20時間平均
```

- **window の解釈は resolution に暗黙的に紐づく**
  - Layer 3（日次データ）: window=20 → 20 bars → 20 days
  - Layer 2（時間データ）: window=20 → 20 bars → 20 hours
- DSL 文法自体は変更不要（bars ベースの計算）
- GP 進化は Layer ごとに独立して動くため、window の意味は自動的に適切になる

### 1.6 alpha-os: Layer 2 Trader

```python
class TacticalTrader:
    """Layer 2: Hourly alpha evaluation for entry/exit timing."""

    def __init__(self, strategic_bias: float, ...):
        self.resolution = "1h"
        self.features = hourly_features  # funding_rate, OI, liquidations
        # GP 進化は Layer 2 専用の feature set で実行

    def run_cycle(self) -> TacticalSignal:
        # 1. Layer 3 の strategic_bias を受け取る
        # 2. hourly features を評価
        # 3. tactical_signal を出力
        #    → strategic_bias と同方向なら実行、逆方向なら待機
        pass
```

### Phase 1 の成果物

- signal-noise に 6-8 本の hourly collector 追加
- alpha-os の 4h サイクルが**異なるシグナルで異なる判断**を毎回出せる
- Layer 2 の GP 進化が funding rate + liquidation パターンを発見
- 取引コスト: 変わらない（エントリータイミングの改善のみ）

### Phase 1 の検証基準

- hourly collector の安定稼働 7 日以上
- Layer 2 alpha の OOS Sharpe > 0 (walk-forward CV on hourly data)
- Layer 3 only vs Layer 3+2 の backtest 比較で改善

---

## Phase 2: Event-Driven Architecture ✅ 完了

**目標**: ポーリングからイベント駆動へ。cron サイクルではなく、
市場イベントがトリガーになる実行モデルへ移行。

### 2.1 signal-noise: Event Bus

```python
# store/events.py
class SignalEvent:
    name: str
    timestamp: str
    value: float
    event_type: str  # "update" | "anomaly" | "circuit_break"

class EventBus:
    """In-process pub/sub for signal updates."""
    async def publish(self, event: SignalEvent) -> None: ...
    async def subscribe(self, pattern: str) -> AsyncIterator[SignalEvent]: ...
```

- `save_collection_result()` が自動的に event を publish
- FastAPI WebSocket endpoint: `/ws/signals?names=funding_rate_btc,liq_ratio_btc`
- In-process `asyncio.Queue` ベース（Redis は不要 — 単一サーバー）

### 2.2 signal-noise: Streaming Collector

```python
class StreamingCollector(BaseCollector):
    """WebSocket-based real-time collector."""

    async def stream(self) -> AsyncIterator[DataFrame]:
        """Yield DataFrames as data arrives."""
        ...
```

- Binance WebSocket (`wss://fstream.binance.com/ws/btcusdt@forceOrder`)
  → リアルタイム清算イベント
- Binance WebSocket (`wss://fstream.binance.com/ws/btcusdt@markPrice`)
  → funding rate 予測値のリアルタイム更新

スケジューラが `stream()` 対応の collector を `asyncio.Task` として起動し、
到着データを即座に store → event bus → consumer に伝播。

### 2.3 alpha-os: Event-Triggered Execution

```python
class EventDrivenTrader:
    """Executes when market conditions warrant, not on fixed schedule."""

    async def watch(self):
        async for event in signal_client.subscribe("funding_rate_*,liq_*"):
            self.buffer.update(event)
            if self._should_evaluate():
                result = self.run_cycle()
                if self._should_execute(result):
                    self.execute(result)

    def _should_evaluate(self) -> bool:
        """Trigger conditions: anomaly, threshold breach, or timer."""
        # funding rate velocity が閾値超過
        # liquidation 急増検出
        # 一定時間（最大 4h）経過
        ...
```

- **通常時**: 4h に 1 回のタイマー評価（現状と同等）
- **イベント時**: funding rate の急変動、大量清算発生時に即座に評価
- **抑制**: 短時間に複数トリガーが来た場合はデバウンス（最小間隔 15 分）

### 2.4 alpha-os: Signal Client の WebSocket 対応

```python
# signal_noise/client.py に追加
class SignalClient:
    async def subscribe(self, pattern: str) -> AsyncIterator[dict]:
        """Subscribe to real-time signal updates via WebSocket."""
        async with websockets.connect(f"{self.ws_url}/ws/signals?names={pattern}"):
            ...
```

### Phase 2 の成果物

- signal-noise が push 通知をサポート（WebSocket）
- alpha-os が市場イベントに即時反応
- cron → asyncio long-running process への移行
- 「大量清算発生 → 15分以内にリバランス」が可能

### Phase 2 の検証基準

- WebSocket 接続の安定性（24h 連続接続、自動再接続）
- イベント到着から注文送信までのレイテンシ < 30 秒
- イベント駆動 vs 定期実行の P&L 比較

---

## Phase 3: Microstructure Layer — 執行最適化 ✅ 完了

**目標**: Layer 1（分足レベル）の情報で執行品質を改善。
alpha の方向性は Layer 2/3 が決める。Layer 1 は「いつ・どう執行するか」に集中。

### 3.1 signal-noise: Orderbook Collector

```python
class BinanceOrderbookCollector(StreamingCollector):
    """Real-time orderbook depth snapshots."""
    # wss://fstream.binance.com/ws/btcusdt@depth20@100ms

    meta = CollectorMeta(
        name="orderbook_btc",
        domain="financial",
        category="microstructure",
        update_frequency="realtime",
        signal_type="orderbook",  # new signal type
        collect_interval=60,  # 1-minute aggregated snapshots stored
    )
```

- 生の orderbook は保存しない（100ms × 24h = 膨大）
- 1 分ごとに集約した derived signals を保存:
  - `book_imbalance_btc`: (bid_volume - ask_volume) / total_volume
  - `book_depth_ratio_btc`: bid_depth / ask_depth (top 5 levels)
  - `spread_bps_btc`: (best_ask - best_bid) / mid_price × 10000

### 3.2 signal-noise: Trade Flow Collector

```python
class BinanceTradeFlowCollector(StreamingCollector):
    """Aggregated trade flow from WebSocket."""
    # wss://fstream.binance.com/ws/btcusdt@aggTrade

    # 1-minute aggregation:
    signals:
      - trade_flow_btc: net buy volume - net sell volume (taker side)
      - vpin_btc: Volume-synchronized Probability of Informed Trading
      - large_trade_count_btc: trades > $100k in 1-min window
```

- VPIN の計算: 直近 50 volume buckets の buy/sell 分類から算出
- 学術的に documented な指標（Easley, López de Prado, O'Hara 2012）

### 3.3 signal-noise: Storage の拡張

高頻度データ用の分離テーブル:

```sql
CREATE TABLE signals_realtime (
    name      TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    value     REAL,
    PRIMARY KEY (name, timestamp)
);

-- Retention policy: 30 days for realtime, unlimited for daily
-- Daily rollup job: signals_realtime → signals (aggregated)
```

- 日次ロールアップで `signals` テーブルに集約
  - `book_imbalance_btc` の日足 = 24h の平均 imbalance
- SQLite のまま（30日 × 1440分 × 10 signals ≈ 432k rows — 問題ない）

### 3.4 alpha-os: Execution Optimizer

```python
class ExecutionOptimizer:
    """Layer 1: Optimize order execution using microstructure signals."""

    def optimal_execution_window(self, side: str) -> bool:
        """Should we execute now or wait?"""
        imbalance = self.get_signal("book_imbalance_btc")
        vpin = self.get_signal("vpin_btc")
        spread = self.get_signal("spread_bps_btc")

        if side == "buy":
            # Buy when: bid-heavy imbalance, low VPIN, tight spread
            return imbalance > 0.1 and vpin < 0.5 and spread < 5.0
        else:
            # Sell when: ask-heavy imbalance, low VPIN, tight spread
            return imbalance < -0.1 and vpin < 0.5 and spread < 5.0

    def split_order(self, total_qty: float) -> list[float]:
        """TWAP/VWAP-style order splitting."""
        ...
```

- Layer 2/3 が「BTC を $5000 分買う」と決定
- Layer 1 が「今は VPIN が高い（informed trader 活動中）から 5 分待つ」と判断
- 結果: slippage の削減、adverse selection の回避

### 3.5 alpha-os: DSL の microstructure 拡張

Layer 1 専用の GP 進化で、microstructure alpha を発見:

```
# Example evolved expressions:
(if_gt vpin_btc 0.7 1.0 0.0)           # VPIN spike → delay
(sub book_imbalance_btc (mean_5 book_imbalance_btc))  # imbalance momentum
(if_gt spread_bps_btc 10.0 0.0 1.0)    # wide spread → don't execute
```

### Phase 3 の成果物

- slippage の体系的な削減（目標: 平均 30% 減）
- adverse selection の回避（VPIN ベース）
- 注文分割の自動化（TWAP with microstructure awareness）

### Phase 3 の検証基準

- slippage: Phase 2 比で 30%+ 削減
- fill rate の維持（タイミング待ちで機会損失しない）
- VPIN 高値での執行回避が事後的に正しかった割合 > 60%

---

## Phase 4: Options Intelligence

**目標**: オプション市場のシグナルを取り込む。
スマートマネーの方向性ベットを読む。

### 4.1 signal-noise: Options Collector

```
collector: deribit_options
domain: financial
category: crypto_derivatives
frequency: hourly
source: Deribit API (public, L1)
signals:
  - iv_atm_btc_7d: 7日 ATM implied volatility
  - iv_atm_btc_30d: 30日 ATM implied volatility
  - iv_skew_btc_7d: 25-delta put IV - 25-delta call IV
  - put_call_ratio_btc: put volume / call volume
  - max_pain_btc: maximum pain price (most options expire worthless)
  - gamma_exposure_btc: dealer gamma exposure (net positioning)
```

- Deribit は BTC options の支配的取引所
- IV surface の変化は「保険のコスト」→ 不確実性の先行指標
- IV skew の急変は方向性ベットの集中を示す

### 4.2 alpha-os: IV-aware Risk Management

- IV 急騰時にポジションサイズを自動縮小
- IV term structure（7d vs 30d）の逆転はイベントリスクの先行指標
- max_pain 付近での mean reversion シグナル

---

## Phase 5: Cross-Exchange Intelligence

**目標**: 複数取引所間の情報格差を活用。

### 5.1 signal-noise: Multi-Exchange Collector

```
collector: cross_exchange_spread
domain: financial
category: microstructure
frequency: realtime (1-min aggregation)
source: CCXT (Binance, Bybit, OKX, Coinbase)
signals:
  - spread_binance_bybit_btc: price difference
  - volume_dominance_btc: Binance volume / total volume
  - lead_lag_btc: which exchange moves first (Granger causality, rolling)
```

### 5.2 alpha-os: Cross-Exchange Alpha

- leader exchange の動きを follower で執行
- volume dominance の shift は regime change の先行指標
- spread の拡大はストレスイベントの検出

---

## Architecture: Final State

```
signal-noise                          alpha-os
┌─────────────────────┐              ┌──────────────────────────────┐
│ Streaming Collectors │              │ Layer 3: Strategic (Daily)   │
│  ├ orderbook WS      │   events    │  ├ macro, on-chain, ETF flow │
│  ├ trade flow WS     ├────────────►│  ├ GP evolution (日次 DSL)   │
│  ├ liquidation WS    │   WebSocket │  └ output: direction bias    │
│  └ funding rate WS   │              │              │               │
│                      │              │              ▼               │
│ Polling Collectors   │              │ Layer 2: Tactical (Hourly)  │
│  ├ OI, LS ratio      │   REST API  │  ├ funding, OI, liquidation │
│  ├ CEX netflow       ├────────────►│  ├ GP evolution (時間 DSL)   │
│  ├ stablecoin supply │              │  └ output: timing signal    │
│  ├ options IV/skew   │              │              │               │
│  └ 1200+ daily sigs  │              │              ▼               │
│                      │              │ Layer 1: Execution (Minute) │
│ Event Bus            │              │  ├ orderbook, VPIN, spread  │
│  ├ signal_updated    │              │  ├ GP evolution (分足 DSL)   │
│  ├ anomaly_detected  │              │  └ output: execute now/wait │
│  └ circuit_broken    │              │              │               │
│                      │              │              ▼               │
│ Storage              │              │ BinanceExecutor              │
│  ├ signals (daily)   │              │  ├ order splitting          │
│  ├ signals_rt (1min) │              │  └ slippage tracking        │
│  └ audit_log         │              └──────────────────────────────┘
└─────────────────────┘
```

## Data Source Priority Matrix

| Source | Layer | Half-Life | API Cost | Infra Cost | Alpha Potential | Priority |
|--------|-------|-----------|----------|------------|----------------|----------|
| Funding rate | L2 | hours | Free (L1) | Low | High | **P1** |
| Liquidation data | L2 | min-hours | Free (L1) | Low | High | **P1** |
| Open interest | L2 | hours | Free (L1) | Low | Medium-High | **P1** |
| Long/short ratio | L2 | hours | Free (L1) | Low | Medium | **P1** |
| CEX netflow | L2 | hours-day | Paid (L2) | Low | High | P2 |
| Stablecoin supply | L2 | hours-day | Free (L1) | Low | Medium | P2 |
| Options IV/skew | L2 | hours | Free (L1) | Low | High | P2 |
| Orderbook imbalance | L1 | minutes | Free (WS) | Medium | Very High | P3 |
| VPIN | L1 | minutes | Free (WS) | Medium | Very High | P3 |
| Trade flow | L1 | minutes | Free (WS) | Medium | High | P3 |
| Cross-exchange spread | L1 | minutes | Free (WS) | Medium | Medium | P4 |

## Implementation Order

```
Phase 1 ✅ 完了 (2026-02)
├── signal-noise: funding rate, liquidation, OI, LS ratio collectors
├── signal-noise: cross-exchange, options signals (hourly)
├── alpha-os: DataStore subdaily support (resolution param)
├── alpha-os: Layer 2 feature set (17 hourly signals) + TacticalTrader
└── alpha-os: validate/forward/paper の L2 対応

Phase 2 ✅ 完了 (2026-03)
├── signal-noise: EventBus (asyncio Queue)
├── signal-noise: WebSocket endpoint (/ws/signals)
├── signal-noise: StreamingCollector base class
├── signal-noise: Binance WS collectors (liquidation, funding rate)
├── alpha-os: SignalClient WS subscribe
└── alpha-os: EventDrivenTrader (デバウンス + 4h フォールバック)

Phase 3 ✅ 完了 (2026-03)
├── signal-noise: signals_realtime table + 30日保持 + 日次ロールアップ
├── signal-noise: BinanceOrderbookCollector (imbalance, depth_ratio, spread)
├── signal-noise: BinanceTradeFlowCollector + VPINCalculator
├── signal-noise: realtime API + CLI rollup + microstructure category
├── alpha-os: ExecutionOptimizer (VPIN/spread/imbalance ベース)
├── alpha-os: BinanceExecutor optimizer 連携
└── alpha-os: MICROSTRUCTURE_SIGNALS + DSL templates (6 seed expressions)

Phase 4 (未着手)
├── signal-noise: Deribit options collector
├── alpha-os: IV-aware risk scaling
└── Validation: risk-adjusted return improvement

Phase 5 (未着手)
├── signal-noise: multi-exchange collectors
├── alpha-os: cross-exchange alpha
└── Validation: lead-lag capture rate
```

## Risk & Constraints

### Technical
- **SQLite の限界**: 1-min データが 100+ signals に達すると書き込み競合。
  → signals_realtime テーブル分離 + batch insert で対応。
  → それでも限界なら TimescaleDB/DuckDB へ移行。
- **WebSocket 安定性**: Binance WS は 24h で切断される仕様。自動再接続必須。
- **Single server**: 現状 CX23 (2 vCPU, 4GB) で全コンポーネントが同居。
  → Phase 3 以降は streaming 処理の CPU 負荷を監視。

### 市場構造
- **Funding rate alpha の寿命**: 広く知られた指標のため、単体でのエッジは限定的。
  GP が他シグナルとの**組み合わせ**で非自明なパターンを発見するのが本質。
- **Microstructure alpha の競争**: HFT firm との直接競争は避ける。
  目標は「最速」ではなく「数分スケールでの賢い執行」。
- **Testnet と本番の乖離**: testnet の orderbook は薄く、
  microstructure シグナルの検証は本番データで行う必要がある。

### 運用
- **データ品質の監視**: hourly collector は日次より failure 頻度が高い。
  signal-noise の circuit breaker + anomaly detection を活用。
- **コスト管理**: Phase 2 以降の paid API（Coinglass 等）は月額が発生。
  → 無料ソースを優先し、paid は alpha 証明後に追加。
