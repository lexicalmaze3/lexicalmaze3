# Yanai

I build RL trading systems that run on real money — not just Kaggle notebooks.

My bot has been walk-forward validated (Sharpe ≥ 0.70 across 5 out-of-sample 
splits) and is currently in live paper trading. Every project below either 
feeds into it or stress-tests an assumption behind it.

---

## What I've built

**[RL Trading Bot](https://github.com/lexicalmaze3/vibe-coded-long-term-stocks-bot)**  
PPO agent (Stable-Baselines3) trading 14 tickers including US equities and 
Israeli defense stocks. Walk-forward validated. Regime-aware position sizing. 
FinBERT news sentiment + Reddit WSB integrated. Currently paper trading.

**[Market Regime Detector](https://github.com/lexicalmaze3/market-regime-detector)**  
HMM + GMM on 26 years of SPY. Correctly identifies 2008, 2020, 2022 as Bear 
Volatile without being told. 68.3% HMM/GMM agreement. Runs live inside the 
trading bot — goes to cash in Bear Volatile, reduces size in Choppy.

**[Volatility Forecasting](https://github.com/lexicalmaze3/volatility-forecasting-ml)**  
GARCH vs LSTM vs Transformer on SPY, TSLA, GLD. Deep models cut RMSE by 40–45% 
vs GARCH. TSLA correlation: 0.50 (GARCH) → 0.83 (Transformer). The interesting 
result: LSTM and Transformer tie — attention adds no benefit over sequential 
memory for 30-day vol windows.

**[Options Pricing](https://github.com/lexicalmaze3/options-pricing-ml)**  
MLP and Transformer pricing 500k synthetic European + American options. 
MLP achieves R²=0.9994. Interesting result: MLP beats Transformer on tabular 
data — bigger model, worse performance. Early exercise premium learned equally 
well for American vs European (MAE $0.668 vs $0.686).

**[Pairs Trading RL](https://github.com/lexicalmaze3/pairs-trading-rl)**  
Cointegration screener + PPO agent trading the spread between equity pairs. 
Walk-forward across 14 folds (2023–2026). Avg OOS Sharpe 3.50. Discovered and 
documented a cointegration regime collapse in 2025–2026.

**[Multi-Agent Market Simulation](https://github.com/lexicalmaze3/multi-agent-market-sim)**  
4 competing RL agents (market maker, momentum trader, arbitrageur, noise trader) 
in a custom limit order book with real SPY price anchoring. Role-specific rewards 
produce emergent behavior — 1,249 trades/episode, market maker profitable via 
spread capture.

**[Earnings Surprise Predictor](https://github.com/lexicalmaze3/earnings-surprise-predictor)**  
Predicts post-earnings drift direction using fundamental + sentiment features. 
Built to stress-test assumptions in the main trading bot.

---

## What makes this different from most ML portfolios

Most people train on clean datasets and report val accuracy.  
These projects are connected — regime detection gates the trading bot,  
vol forecasting will gate position sizing, options pricing stress-tests  
derivative assumptions. The goal is a system, not a collection of models.

---

## Stack
Python · PyTorch · Stable-Baselines3 · hmmlearn · arch · FinBERT · yfinance

## Currently
Paper trading. 4-week validation window before real capital.
