# Yanai

I'm 16, self-taught, and I build things because watching them come to life is genuinely fun.

Right now that means RL trading systems — walk-forward validated, paper trading, moving to real capital after a 4-week validation window. Every project below either feeds into the main bot or stress-tests an assumption behind it.

---

## What I've built

**[RL Trading Bot](https://github.com/lexicalmaze3/vibe-coded-long-term-stocks-bot)**  
PPO agent (Stable-Baselines3) trading 14 tickers including US equities and Israeli defense stocks. Walk-forward validated (Sharpe ≥ 0.70 across 5 out-of-sample splits). Regime-aware position sizing. FinBERT news sentiment integrated. Currently paper trading.

**[Multi-Agent Market Simulation](https://github.com/lexicalmaze3/multi-agent-market-sim)**  
4 competing RL agents (market maker, momentum trader, arbitrageur, noise trader) in a custom limit order book anchored to real SPY prices. Role-specific rewards produce emergent behavior — 1,249 trades/episode, market maker profitable via spread capture. Nash equilibrium collapse discovered and fixed mid-build.

**[Pairs Trading RL](https://github.com/lexicalmaze3/pairs-trading-rl)**  
Cointegration screener + PPO agent trading equity pair spreads. Walk-forward across 14 folds (2023–2026). Avg OOS Sharpe 3.50. Documented a full cointegration regime collapse in 2025–2026 — the pair was real in training and dead in production.

**[Market Regime Detector](https://github.com/lexicalmaze3/market-regime-detector)**  
HMM + GMM on 26 years of SPY. Correctly identifies 2008, 2020, 2022 as Bear Volatile without being told. 68.3% agreement between models. Runs live inside the trading bot.

**[Volatility Forecasting](https://github.com/lexicalmaze3/volatility-forecasting-ml)**  
GARCH vs LSTM vs Transformer on SPY, TSLA, GLD. Deep models cut RMSE by 40–45% vs GARCH. Interesting result: LSTM and Transformer tie — attention adds no benefit over sequential memory for 30-day vol windows.

**[Options Pricing](https://github.com/lexicalmaze3/options-pricing-ml)**  
MLP and Transformer pricing 500k synthetic European + American options. MLP achieves R²=0.9994. Bigger model, worse performance — the result was more interesting than a clean win.

**[Earnings Surprise Predictor](https://github.com/lexicalmaze3/earnings-surprise-predictor)**  
Predicts post-earnings drift direction using fundamental + sentiment features. Built to stress-test assumptions in the main bot.

---

## What makes this different from most ML portfolios

Most people train on clean datasets and report val accuracy.  
These projects are connected — regime detection gates the trading bot, vol forecasting will gate position sizing, options pricing stress-tests derivative assumptions. The goal is a system, not a collection of models.

---

## Currently building

Multi-agent market sim Phase 2 — self-play training so agents adapt to each other over time.

---

## Stack

Python · PyTorch · Stable-Baselines3 · PettingZoo · hmmlearn · arch · FinBERT · yfinance

---

## Paper trading positions open since 2026-05-21

SPY · TSLA · RTX · NOC · ESLT · GLD
