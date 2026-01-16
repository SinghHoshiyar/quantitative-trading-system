# Quantitative Trading System - Methodology

## Overview

This document explains the theoretical foundations, mathematical models, and design decisions behind the quantitative trading system.

---

## 1. Data Acquisition & Engineering

### 1.1 Market Data Selection

**NIFTY 50 Index**
- Most liquid Indian equity index
- High trading volume ensures tight spreads
- Represents broad market exposure

**5-Minute Timeframe**
- Balance between noise and signal
- Sufficient data points for statistical analysis
- Practical for intraday strategies

**Derivatives Coverage**
- **Futures**: Price discovery, basis trading
- **Options**: Volatility information, market sentiment
- **ATM ± 2 Strikes**: Captures relevant price action without excessive data

### 1.2 Data Quality Assurance

**Missing Data Handling**
```
Method: Forward Fill + Backward Fill
Rationale: Preserves last known price during low-liquidity periods
Alternative: Linear interpolation (not used to avoid artificial smoothing)
```

**Outlier Detection**
```
Method: Z-score with 5-sigma threshold
Formula: z = (x - μ) / σ
Rationale: Removes bad ticks while preserving genuine volatility spikes
```

**Timestamp Alignment**
```
Method: Merge on exact timestamps with forward fill
Rationale: Ensures all instruments are synchronized
Challenge: Handles different trading hours and holidays
```

---

## 2. Feature Engineering

### 2.1 Technical Indicators

**Exponential Moving Averages (EMA)**
```
Formula: EMA_t = α × Price_t + (1 - α) × EMA_{t-1}
Where: α = 2 / (period + 1)

EMA(5): Fast-moving, captures short-term trends
EMA(15): Slower, filters noise

Crossover Signal:
- Bullish: EMA(5) > EMA(15)
- Bearish: EMA(5) < EMA(15)
```

**Relative Strength Index (RSI)**
```
RSI = 100 - (100 / (1 + RS))
Where: RS = Average Gain / Average Loss over 14 periods

Interpretation:
- RSI > 70: Overbought
- RSI < 30: Oversold
```

**Bollinger Bands**
```
Middle Band = 20-period SMA
Upper Band = Middle + (2 × σ)
Lower Band = Middle - (2 × σ)

BB Position = (Price - Lower) / (Upper - Lower)
```

### 2.2 Options Greeks (Black-Scholes Model)

**Assumptions**
- European-style options (NIFTY options are European)
- Constant risk-free rate (6.5% - RBI repo rate)
- Log-normal price distribution
- No dividends (index options)

**Delta (Δ)**
```
Call: Δ_c = N(d1)
Put: Δ_p = N(d1) - 1

Where:
d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)

Interpretation: Rate of change of option price w.r.t. underlying
Range: Call [0, 1], Put [-1, 0]
```

**Gamma (Γ)**
```
Γ = N'(d1) / (S × σ × √T)

Where: N'(x) = (1/√(2π)) × e^(-x²/2)

Interpretation: Rate of change of delta
Highest at ATM, decreases away from ATM
```

**Theta (Θ)**
```
Call: Θ_c = -[S × N'(d1) × σ / (2√T)] - r × K × e^(-rT) × N(d2)
Put: Θ_p = -[S × N'(d1) × σ / (2√T)] + r × K × e^(-rT) × N(-d2)

Interpretation: Time decay (per day)
Always negative for long options
```

**Vega (ν)**
```
ν = S × √T × N'(d1) / 100

Interpretation: Sensitivity to 1% change in IV
Same for calls and puts
Highest at ATM
```

**Rho (ρ)**
```
Call: ρ_c = K × T × e^(-rT) × N(d2) / 100
Put: ρ_p = -K × T × e^(-rT) × N(-d2) / 100

Interpretation: Sensitivity to 1% change in interest rate
Less important for short-dated options
```

### 2.3 Derived Features

**Implied Volatility (IV) Metrics**
```
IV Skew = IV_put - IV_call
Interpretation: Market fear gauge
Positive skew → Put premium > Call premium → Bearish sentiment

IV Average = (IV_call + IV_put) / 2
Interpretation: Overall volatility expectation
```

**Put-Call Ratio (PCR)**
```
PCR_OI = Put Open Interest / Call Open Interest
PCR_Volume = Put Volume / Call Volume

Interpretation:
- PCR > 1: More puts → Bearish sentiment
- PCR < 1: More calls → Bullish sentiment
- Contrarian indicator: Extreme values signal reversals
```

**Futures Basis**
```
Basis = (Futures Price - Spot Price) / Spot Price

Interpretation:
- Positive basis (contango): Normal market
- Negative basis (backwardation): Stress/shortage
- Basis convergence at expiry
```

**Delta Neutrality**
```
Delta Skew = Δ_call + Δ_put

Interpretation:
- Near 0: Market is delta-neutral
- Deviation indicates directional bias
```

**Gamma Exposure**
```
Gamma Exposure = Γ_call + Γ_put

Interpretation:
- High gamma → Large delta changes → Volatility
- Dealers hedge gamma → Price pinning near strikes
```

---

## 3. Regime Detection

### 3.1 Hidden Markov Model (HMM)

**Theoretical Foundation**

HMM assumes:
1. Market exists in hidden states (regimes)
2. Observable features depend on hidden state
3. State transitions follow Markov property

**Model Specification**
```
States: S = {Downtrend, Sideways, Uptrend}
Observations: O = Options-based features

Transition Matrix A:
A_ij = P(state_t = j | state_{t-1} = i)

Emission Probabilities B:
B_j(o) = P(observation = o | state = j)

Initial Probabilities π:
π_i = P(state_0 = i)
```

**Training Algorithm: Baum-Welch (EM)**
```
E-step: Calculate forward-backward probabilities
M-step: Update parameters (A, B, π)
Iterate until convergence
```

**Inference: Viterbi Algorithm**
```
Find most likely state sequence given observations
Dynamic programming approach
O(T × N²) complexity
```

### 3.2 Feature Selection for Regimes

**Why Options-Based Features?**

1. **Forward-Looking**: Options embed market expectations
2. **Sentiment**: PCR captures fear/greed
3. **Volatility**: IV predicts future turbulence
4. **Structural**: Greeks reveal market positioning

**Selected Features**
- IV (call, put): Volatility regime
- PCR (OI, volume): Sentiment regime
- Futures basis: Market structure
- Delta skew: Directional bias
- Gamma exposure: Volatility clustering

### 3.3 Regime Interpretation

**Uptrend (State 2)**
- Positive average returns
- Lower IV (complacency)
- Low PCR (bullish sentiment)
- Positive futures basis

**Sideways (State 1)**
- Near-zero returns
- Moderate IV
- Balanced PCR
- Narrow trading range

**Downtrend (State 0)**
- Negative returns
- High IV (fear)
- High PCR (hedging demand)
- Potential backwardation

---

## 4. Trading Strategy

### 4.1 Signal Generation

**EMA Crossover**
```
Long Signal: EMA(5) crosses above EMA(15)
Short Signal: EMA(5) crosses below EMA(15)

Rationale: Trend-following, momentum-based
Lag: ~2-3 periods (acceptable for 5-min data)
```

**Regime Filter**
```
Allow Long: Only in Uptrend regime
Allow Short: Only in Downtrend regime
No Trading: Sideways regime

Rationale: Trade with the regime, not against it
Reduces false signals in choppy markets
```

### 4.2 Risk Management

**Position Sizing**
```
Position Size = Capital × Risk_per_Trade / Entry_Price
Risk_per_Trade = 2% (conservative)

Rationale: Kelly Criterion suggests optimal f ≈ edge/odds
Conservative approach for risk management
```

**Stop Loss**
```
Stop Loss = 2% from entry price

Rationale:
- Limits maximum loss per trade
- Based on average true range (ATR)
- Prevents catastrophic losses
```

**Take Profit**
```
Take Profit = 4% from entry price
Risk-Reward Ratio = 2:1

Rationale:
- Positive expectancy even with 40% win rate
- Locks in profits before reversals
```

### 4.3 Backtest Methodology

**Walk-Forward Approach**
```
1. Train on historical data
2. Test on out-of-sample period
3. No look-ahead bias
4. Realistic execution assumptions
```

**Performance Metrics**

**Sharpe Ratio**
```
Sharpe = (R_p - R_f) / σ_p

Where:
R_p = Portfolio return
R_f = Risk-free rate (6.5%)
σ_p = Portfolio volatility

Interpretation:
> 1.0: Good
> 2.0: Excellent
> 3.0: Exceptional
```

**Sortino Ratio**
```
Sortino = (R_p - R_f) / σ_downside

Only penalizes downside volatility
Better for asymmetric returns
```

**Calmar Ratio**
```
Calmar = Annual Return / Max Drawdown

Measures return per unit of worst-case risk
Higher is better
```

**Maximum Drawdown**
```
MDD = max(Peak - Trough) / Peak

Worst peak-to-trough decline
Critical for risk assessment
```

---

## 5. Machine Learning Enhancement

### 5.1 Problem Formulation

**Binary Classification**
```
Target: Will next trade be profitable?
Y = 1 if future_return > 0
Y = 0 otherwise

Look-ahead: 20 periods (100 minutes)
```

**Feature Engineering for ML**
- All technical indicators
- All Greeks
- All derived features
- Current regime
- Lagged features (optional)

### 5.2 XGBoost Model

**Algorithm: Gradient Boosting Decision Trees**
```
Objective: Minimize loss function
L(θ) = Σ l(y_i, ŷ_i) + Σ Ω(f_k)

Where:
l = Loss function (log loss for classification)
Ω = Regularization term (L1 + L2)
f_k = Individual trees
```

**Hyperparameters**
```
n_estimators: 200 (number of trees)
max_depth: 6 (tree depth, prevents overfitting)
learning_rate: 0.05 (shrinkage, improves generalization)
subsample: 0.8 (row sampling, reduces overfitting)
colsample_bytree: 0.8 (column sampling, feature diversity)
```

**Advantages**
- Handles non-linear relationships
- Feature importance built-in
- Robust to outliers
- Fast training and prediction

### 5.3 LSTM Model

**Architecture: Recurrent Neural Network**
```
Input: Sequence of features (20 timesteps)
LSTM Layer 1: 64 units, return sequences
Dropout: 0.2 (regularization)
LSTM Layer 2: 32 units
Dropout: 0.2
Dense Output: 1 unit, sigmoid activation
```

**Why LSTM?**
- Captures temporal dependencies
- Remembers long-term patterns
- Handles sequential market data
- Learns complex non-linear dynamics

**Training**
```
Loss: Binary crossentropy
Optimizer: Adam (adaptive learning rate)
Batch size: 32
Epochs: 50 with early stopping
```

### 5.4 Model Evaluation

**Metrics**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 Score = 2 × (Precision × Recall) / (Precision + Recall)
AUC-ROC = Area under ROC curve
```

**Cross-Validation**
```
Time-series split (no shuffling)
Train: 70%
Validation: 15%
Test: 15%
```

---

## 6. Outlier Analysis

### 6.1 Statistical Methodology

**Z-Score Method**
```
z = (x - μ) / σ

Outlier if |z| > 3 (3-sigma rule)
Captures ~99.7% of normal distribution
```

**Why 3-Sigma?**
- Balance between sensitivity and specificity
- Captures truly exceptional trades
- Reduces false positives

### 6.2 Pattern Recognition

**Comparative Analysis**
```
Compare outliers vs normal trades on:
1. Regime distribution
2. Time of day
3. IV environment
4. Duration
5. Entry conditions
```

**Statistical Tests**
```
T-test: Compare means (e.g., IV levels)
Chi-square: Compare distributions (e.g., regimes)
Mann-Whitney U: Non-parametric alternative
```

### 6.3 Insight Generation

**Causal vs Correlational**
- Identify correlations
- Avoid causal claims without evidence
- Use insights for hypothesis generation
- Validate with additional testing

---

## 7. Design Decisions & Rationale

### 7.1 Why This Approach?

**Regime-Based Trading**
- Markets are non-stationary
- Different strategies work in different regimes
- Reduces false signals in choppy markets

**Options Integration**
- Forward-looking information
- Captures market sentiment
- Provides volatility insights
- Enhances feature set

**ML Enhancement**
- Improves trade quality, not quantity
- Learns complex patterns
- Adapts to market changes
- Provides confidence scores

### 7.2 Limitations & Assumptions

**Assumptions**
1. Historical patterns persist
2. Market microstructure is stable
3. No significant regime changes
4. Execution at mid-price (no slippage)
5. No transaction costs (can be added)

**Limitations**
1. Synthetic data for demonstration
2. No live execution
3. Simplified risk management
4. No portfolio optimization
5. Single-asset focus

### 7.3 Production Considerations

**For Live Trading**
1. Real-time data feeds
2. Order execution system
3. Risk management layer
4. Position monitoring
5. Performance tracking
6. Regulatory compliance
7. Disaster recovery

---

## 8. References

### Academic Papers
1. Baum, L. E., & Petrie, T. (1966). Statistical inference for probabilistic functions of finite state Markov chains.
2. Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities.
3. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
4. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory.

### Books
1. "Advances in Financial Machine Learning" - Marcos López de Prado
2. "Algorithmic Trading" - Ernest P. Chan
3. "Options, Futures, and Other Derivatives" - John C. Hull
4. "Machine Learning for Asset Managers" - Marcos López de Prado

### Libraries & Tools
1. pandas, numpy - Data manipulation
2. scikit-learn - ML utilities
3. XGBoost - Gradient boosting
4. TensorFlow/Keras - Deep learning
5. hmmlearn - Hidden Markov Models
6. scipy - Statistical functions

---

## Conclusion

This methodology combines:
- **Statistical rigor**: HMM, hypothesis testing
- **Financial theory**: Options pricing, risk management
- **Machine learning**: XGBoost, LSTM
- **Engineering discipline**: Clean code, modularity

The result is a professional, production-quality quantitative trading system suitable for demonstrating skills in ML engineering and quantitative research roles.
