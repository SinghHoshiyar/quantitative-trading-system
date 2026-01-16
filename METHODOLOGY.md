# Methodology

## 1. Data Pipeline

### 1.1 Data Acquisition
The system processes three types of market data:

**NIFTY 50 Spot (^NSEI)**
- 5-minute OHLCV bars
- 1 year historical data
- Primary price reference

**NIFTY Futures**
- Current month contract
- Automatic rollover handling on expiry
- Open Interest tracking
- Basis calculation vs spot

**NIFTY Options Chain**
- ATM ± 2 strikes (5 strikes total)
- Both Call and Put options
- Implied Volatility (IV)
- Open Interest and Volume
- Last Traded Price (LTP)

### 1.2 Data Cleaning
- Missing data: Forward-fill followed by backward-fill
- Outlier removal: Z-score method (5-sigma threshold)
- Timestamp alignment: Exact match with forward-fill for gaps
- Futures rollover: Seamless transition on expiry dates
- ATM calculation: Dynamic strike selection based on spot price

## 2. Feature Engineering

### 2.1 Technical Indicators
- **EMA (5, 15)**: Exponential Moving Averages for trend detection
- **RSI (14)**: Relative Strength Index for momentum
- **Bollinger Bands (20, 2)**: Volatility bands
- **Volume indicators**: Volume ratios and moving averages

### 2.2 Options Greeks (Black-Scholes)
Calculated for all strikes (ATM ± 2) for both Calls and Puts:

**Delta (Δ)**: Rate of change of option price w.r.t. underlying
- Call: 0 to 1
- Put: -1 to 0

**Gamma (Γ)**: Rate of change of Delta
- Highest at ATM
- Measures convexity

**Theta (Θ)**: Time decay per day
- Always negative for long positions
- Accelerates near expiry

**Vega (ν)**: Sensitivity to 1% IV change
- Highest at ATM
- Same for Calls and Puts

**Rho (ρ)**: Sensitivity to 1% interest rate change
- Less significant for short-dated options

**Assumptions**:
- Risk-free rate: 6.5% (RBI repo rate)
- European-style options
- No dividends (index options)
- Log-normal price distribution

### 2.3 Derived Features
- **IV Metrics**: Average IV, IV skew (Put - Call)
- **Put-Call Ratio**: Based on OI and Volume
- **Futures Basis**: (Futures - Spot) / Spot
- **Delta Skew**: Sum of Call and Put deltas
- **Gamma Exposure**: Total gamma across strikes

Total features: 50+

## 3. Regime Detection

### 3.1 Hidden Markov Model (HMM)
**States**: 3 (Downtrend, Sideways, Uptrend)

**Input Features** (options-based only):
- IV (ATM Call, ATM Put)
- Put-Call Ratio (OI, Volume)
- Futures Basis
- Delta Skew
- Gamma Exposure

**Training**: Baum-Welch algorithm (Expectation-Maximization)

**Inference**: Viterbi algorithm for most likely state sequence

**Rationale**: Options data is forward-looking and captures market sentiment better than price alone.

### 3.2 Regime Characteristics
- **Uptrend**: Positive returns, lower IV, bullish sentiment
- **Sideways**: Range-bound, moderate IV, balanced sentiment
- **Downtrend**: Negative returns, high IV (fear), bearish sentiment

## 4. Trading Strategy

### 4.1 Signal Generation
**Base Signal**: EMA(5) × EMA(15) crossover
- Long: EMA(5) crosses above EMA(15)
- Short: EMA(5) crosses below EMA(15)

**Regime Filter**:
- Long trades: Only in Uptrend regime
- Short trades: Only in Downtrend regime
- No trading: Sideways regime

### 4.2 Risk Management
- **Position Size**: 2% of capital per trade
- **Stop Loss**: 2% from entry
- **Take Profit**: 4% from entry
- **Risk-Reward Ratio**: 1:2

### 4.3 Backtest Methodology
- Walk-forward approach (no look-ahead bias)
- Realistic execution assumptions
- No transaction costs (can be added)
- Performance metrics: Sharpe, Sortino, Calmar, Max Drawdown

## 5. Machine Learning

### 5.1 Problem Formulation
**Task**: Binary classification - Will the next trade be profitable?

**Target**: 1 if future return > 0, else 0

**Features**: All technical indicators + Greeks + derived features + regime

**Split**: 70% train, 15% validation, 15% test (time-series split)

### 5.2 XGBoost Model
**Algorithm**: Gradient Boosting Decision Trees

**Hyperparameters**:
- Trees: 200
- Max depth: 6
- Learning rate: 0.05
- Subsample: 0.8
- Column sample: 0.8

**Advantages**: Handles non-linearity, provides feature importance, robust to outliers

### 5.3 LSTM Model
**Architecture**:
- Input: 20-timestep sequences
- LSTM Layer 1: 64 units
- Dropout: 0.2
- LSTM Layer 2: 32 units
- Dropout: 0.2
- Dense Output: 1 unit (sigmoid)

**Training**: Binary crossentropy loss, Adam optimizer, early stopping

**Advantages**: Captures temporal dependencies, learns sequential patterns

### 5.4 ML-Enhanced Strategy
Only execute trades when ML confidence > 0.5, improving trade quality over quantity.

## 6. Performance Analysis

### 6.1 Outlier Detection
**Method**: Z-score with 3-sigma threshold

**Purpose**: Identify exceptionally profitable trades

### 6.2 Pattern Recognition
Compare outlier vs normal trades across:
- Regime distribution
- Time of day
- IV environment
- Trade duration
- Entry conditions

### 6.3 Statistical Testing
- T-tests for mean comparisons
- Chi-square for distribution comparisons
- Hypothesis validation

## 7. Key Design Decisions

### Why Regime-Based Trading?
Markets are non-stationary. Different strategies work in different market conditions. Regime filtering reduces false signals in choppy markets.

### Why Options Data?
Options embed forward-looking information and market sentiment. Greeks provide insights into market positioning and expected volatility.

### Why ML Enhancement?
ML learns complex non-linear patterns that rule-based systems miss. Focus on improving trade quality rather than generating more signals.

## 8. Limitations

- Synthetic data used for demonstration
- No transaction costs modeled
- Simplified execution (mid-price fills)
- Single-asset focus
- No portfolio optimization
- Assumes historical patterns persist

## 9. Production Considerations

For live trading, additional requirements:
- Real-time data feeds
- Order execution system
- Slippage and transaction cost modeling
- Position monitoring and alerts
- Risk management layer
- Regulatory compliance
- Disaster recovery procedures
