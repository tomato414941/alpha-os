# Candidate Matrix

This matrix ranks broad profit-source candidates by data reachability,
execution realism, and falsification speed.

It is not a list of approved strategies. It is a way to avoid narrow local
optimization around whichever probe was easiest to script.

## Priority Meaning

- **P0**: most worth parallel work now
- **P1**: promising, but blocked by data, execution, or a larger modeling step
- **P2**: useful context or long-horizon work, not the immediate path

## Matrix

| priority | profit source | hypothesis | required data | reachable provider | access | history | execution venue | holding period | capacity concern | main failure mode | first falsification test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P0 | liquidation/OI/funding cascade | forced flow plus crowded OI creates short-horizon continuation or reversal | liquidation events, OI, funding, perp trades, L2 depth | CoinGlass, Coinalyze, Kwery, exchange streams | likely paid/auth for good history | yes if provider paid; exchange live only may be limited | perps on Binance/Bybit/OKX/Hyperliquid | minutes to hours | disappears with fees/slippage; needs fast execution | public liquidation feeds can be delayed/underreported | get 30d liquidation/OI history and test post-liquidation return by side/size |
| P0 | multi-venue funding/basis carry | funding or basis spread persists long enough to hedge across venues | funding history, spot/perp prices, fees, margin, borrow, venue depth | Hyperliquid, Binance data vision, CoinGlass, Coinalyze, CCXT | mixed free/paid/auth | partial free; better paid | spot/perp or perp/perp across venues | hours to days | capital split, margin, borrow, transfer limits | apparent spread is eaten by fees, borrow, or unavailable legs | collect hourly multi-venue funding history and test hedged net PnL after fees |
| P0 | L2 market making / adverse-selection filter | spread capture works only when near-book imbalance and flow reduce adverse selection | L2 snapshots/diffs, trades, fees, fills | Hyperliquid API, Hummingbot connectors, Kwery, CoinAPI | free current; history likely paid/auth | current free, history weak | Hyperliquid or CEX CLOB | seconds to minutes | queue position and inventory capacity | fills occur only before adverse moves | collect repeated L2/trade snapshots and estimate fill-side next-return |
| P0 | paper-trading executable carry | easiest edge candidate must survive order tickets, fees, and live monitoring | selected carry candidate, venue rules, balances, fees, funding schedule | chosen exchange APIs, CCXT, Hyperliquid | auth required | live/paper only | one selected venue pair | hours to days | account size and margin | strategy is not operationally tradable | produce manual/paper order tickets and compare intended vs actual fills/funding |
| P0 | stablecoin liquidity regime | stablecoin supply contraction/expansion changes risk appetite and funding/returns | stablecoin supply, chain distribution, BTC/ETH/perp returns, funding | DeFiLlama, CoinGecko, CryptoQuant, Glassnode | free snapshot; serious flow likely paid | some free current; history provider-dependent | directional or risk filter across crypto | days to weeks | broad beta, not standalone capacity | supply changes lag price or are non-causal | join stablecoin weekly changes to BTC/ETH/perp returns and funding regimes |
| P1 | attention/news event reaction | attention spikes and fear/greed extremes create tradable lead/lag | trending, news timestamps, social posts, returns, volume | CoinGecko, Alternative.me, paid news/social feeds, X/Reddit APIs | mixed; serious feeds auth/paid | partial | spot/perps by listed asset | hours to days | event crowding and fast decay | attention is lagging price | label trending/news events and test forward returns with timestamp leakage audit |
| P1 | DeFi yield persistence | high APY is tradable only when yield persists and exit liquidity survives | APY history, TVL, reward emissions, withdrawal constraints, gas | DeFiLlama, protocol APIs, Dune | mixed free/paid/auth | partial | DeFi protocols, stablecoins | days to weeks | contract/custody/exit risk | APY decays when capital enters | test APY persistence and TVL inflow/outflow after top-yield snapshots |
| P1 | stablecoin depeg/repeg | peg deviations can mean stress trade or redemption trade | stablecoin price, liquidity, issuer mechanism, redemption route | DeFiLlama, CoinGecko, DEX APIs, CEX order books | mixed | yes for price; liquidity harder | spot/CEX/DEX | minutes to days | redemption/counterparty risk | depeg reflects real insolvency, not mispricing | identify historical depegs and test recovery conditional on liquidity/mechanism |
| P1 | cross-exchange execution arb | fragmented venues create short-lived price differences | top-of-book, fees, balances, transfer, latency | CCXT/CCXT Pro, CoinAPI, Kwery, exchange APIs | auth for execution | live required | CEX/CEX or CEX/DEX | seconds to minutes | balances must exist on both venues | spread not executable after latency/fees | monitor two venues live and count executable spreads after fees |
| P1 | sector rotation | crypto capital rotates across sectors faster than single-asset momentum | sector taxonomy, returns, volume, attention, funding | CoinGecko categories, GeckoTerminal, DefiLlama, custom taxonomy | mostly free for current; history mixed | partial | spot/perps sector baskets | days to weeks | token liquidity and survivorship | taxonomy and universe leak future winners | build fixed historical sector universe and compare sector momentum vs BTC |
| P1 | token unlock/listing events | supply and venue events create predictable pressure or attention | unlock calendar, listing timestamps, float, volume, derivatives | TokenUnlocks, exchange announcements, CoinGecko, paid feeds | likely mixed/paid | yes if provider | spot/perps around event assets | hours to weeks | event crowding, borrow availability | event is priced before signal arrives | create event calendar and test pre/post return without future event leakage |
| P1 | options volatility | IV/skew mispricing or vol-risk premium creates non-directional edge | options chain, IV, Greeks, realized vol, funding | Deribit, CoinGlass, Laevitas | auth/paid likely | yes | Deribit/options venues | days to weeks | options liquidity and margin | retail-size spreads are too wide | test IV-vs-realized and skew carry after bid/ask costs |
| P1 | RL execution/sizing | direct reward optimization may improve sizing/execution over hand rules | simulator, actions, reward, costs, states | internal once simulator exists; external engines possible | internal plus market data | depends on simulator history | whichever strategy becomes executable | variable | overfit and simulator mismatch | policy learns simulator artifacts | build environment for one candidate and compare to simple baseline OOS |
| P2 | macro liquidity overlay | rates/DXY/yields/liquidity help avoid bad regimes | macro data, cross-asset prices, crypto returns | FRED, Yahoo, paid macro feeds | mostly free | yes | risk filter, not standalone | days to months | broad beta only | too slow for crypto timing | test whether macro filters reduce drawdown without killing return |
| P2 | protocol fundamentals | revenue/users/TVL quality explains medium-term token repricing | protocol fees, revenue, users, TVL, valuation | DefiLlama, Token Terminal, Dune | mixed; serious data paid | partial | spot/perps | weeks to months | token liquidity, fundamentals lag | fundamentals do not map to token value | rank protocols by revenue/valuation and test forward returns |
| P2 | bridge/liquidity migration | cross-chain stablecoin/TVL flows precede sector or chain repricing | bridge flows, chain TVL, stablecoin distribution, token returns | DefiLlama, Dune, bridge APIs | mixed | partial | chain tokens / sector baskets | days to weeks | bridge data quality | flow follows price rather than leads | test chain flow changes vs chain-token returns |
| P2 | prediction market signal | event odds provide structured probability features | Polymarket/Kalshi odds, event labels, crypto/macro prices | Polymarket/Kalshi APIs, Kwery | mixed/auth | yes depending provider | directional/event hedges | hours to weeks | market coverage and liquidity | odds are not related to tradable asset | map event odds to related assets and test lead/lag |
| P2 | anomaly/stress detection | abnormal cross-source states identify risk or opportunity | funding, spread, APY, peg, OI, flow, attention | existing probes plus paid sources | mixed | requires collection | risk filter or event trade | minutes to days | many false positives | anomaly is noise or already priced | define anomaly score and test next-period return/drawdown |

## Immediate Parallel Work

Do these in parallel rather than one narrow deep dive:

1. **Derivatives forced-flow lane**
   - Find reachable liquidation + OI history.
   - First falsification: post-liquidation returns by side and size.

2. **Funding/basis executable lane**
   - Turn current funding/basis snapshots into hourly history.
   - First falsification: net hedged PnL after maker/taker fees and borrow/margin assumptions.

3. **L2/fill lane**
   - Collect repeated Hyperliquid L2 snapshots plus trades.
   - First falsification: whether visible imbalance predicts adverse selection or fill-side next return.

4. **Attention/liquidity context lane**
   - Join stablecoin supply, fear/greed, trending, and returns.
   - First falsification: whether context improves returns or drawdowns versus a price-only baseline.

5. **Operational lane**
   - Pick one candidate and produce paper/manual trade tickets.
   - First falsification: whether the trade can be placed, monitored, and reconciled at all.

## Do Not Do Next

- Do not add another one-off current snapshot unless it covers a genuinely new
  profit source.
- Do not tune daily momentum variants before broader data lanes are tested.
- Do not promote new abstractions into `src/alpha_os` until multiple real
  strategy candidates need the same shape.
- Do not call a data route an edge.

