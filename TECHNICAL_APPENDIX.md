# Technical Appendix

## A. Mathematical Foundations

### A.1 Black-Scholes Options Pricing

**Call Option Price:**
```
C = S₀N(d₁) - Ke⁻ʳᵀN(d₂)
```

**Put Option Price:**
```
P = Ke⁻ʳᵀN(-d₂) - S₀N(-d₁)
```

**Where:**
```
d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T

S₀ = Current spot price
K = Strike price
r = Risk-free rate (0.065)
T = Time to expiry (years)
σ = Implied volatility
N(x) = Cumulative standard normal distribution
```

**Greeks Formulas:**

**Delta:**
```
Δ_call = N(d₁)
Δ_put = N(d₁) - 1
```

**Gamma:**
```
Γ = N'(d₁) / (S₀σ√T)
where N'(x) = (1/√(2π)) × e^(-x²/2)
```

**Theta:**
```
Θ_call = -[S₀N'(d₁)σ / (2√T)] - rKe⁻ʳᵀN(d₂)
Θ_put = -[S₀N'(d₁)σ / (2√T)] + rKe⁻ʳᵀN(-d₂)
```

**Vega:**
```
ν = S₀√T × N'(d₁) / 100
```

**Rho:**
```
ρ_call = KTe⁻ʳᵀN(d₂) / 100
ρ_put = -KTe⁻ʳᵀN(-d₂) / 100
```

### A.2 Hidden Markov Model

**Model Definition:**
```
States: S = {s₁, s₂, ..., sₙ}
Observations: O = {o₁, o₂, ..., oₜ}

Transition Matrix A:
A_ij = P(state_t = j | state_{t-1} = i)

Emission Matrix B:
B_j(o) = P(observation = o | state = j)

Initial Probabilities π:
π_i = P(state₀ = i)
```

**Forward Algorithm:**
```
α_t(i) = P(o₁, o₂, ..., o_t, state_t = i | λ)

Initialization:
α₁(i) = π_i × B_i(o₁)

Recursion:
α_{t+1}(j) = [Σᵢ α_t(i) × A_ij] × B_j(o_{t+1})
```

**Viterbi Algorithm:**
```
δ_t(i) = max P(state₁, ..., state_{t-1}, state_t = i, o₁, ..., o_t | λ)

Initialization:
δ₁(i) = π_i × B_i(o₁)
ψ₁(i) = 0

Recursion:
δ_t(j) = max_i [δ_{t-1}(i) × A_ij] × B_j(o_t)
ψ_t(j) = argmax_i [δ_{t-1}(i) × A_ij]

Termination:
P* = max_i δ_T(i)
state_T* = argmax_i δ_T(i)

Backtracking:
state_t* = ψ_{t+1}(state_{t+1}*)
```

### A.3 XGBoost Objective Function

**Objective:**
```
L(θ) = Σᵢ l(yᵢ, ŷᵢ) + Σₖ Ω(fₖ)

where:
l = Loss function (log loss for classification)
Ω(f) = γT + (λ/2)||w||² (regularization)
T = Number of leaves
w = Leaf weights
```

**Gradient Boosting:**
```
ŷᵢ⁽ᵗ⁾ = ŷᵢ⁽ᵗ⁻¹⁾ + η × fₜ(xᵢ)

where:
η = Learning rate
fₜ = New tree at iteration t
```

**Split Finding:**
```
Gain = ½ [(G_L²/(H_L + λ)) + (G_R²/(H_R + λ)) - ((G_L + G_R)²/(H_L + H_R + λ))] - γ

where:
G = Σᵢ gᵢ (gradient)
H = Σᵢ hᵢ (hessian)
L, R = Left, Right child nodes
```

### A.4 LSTM Cell Equations

**Forget Gate:**
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
```

**Input Gate:**
```
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
```

**Cell State Update:**
```
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
```

**Output Gate:**
```
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
h_t = o_t ⊙ tanh(C_t)
```

**Where:**
```
σ = Sigmoid activation
⊙ = Element-wise multiplication
W = Weight matrices
b = Bias vectors
```

## B. Performance Metrics

### B.1 Risk-Adjusted Returns

**Sharpe Ratio:**
```
Sharpe = (R_p - R_f) / σ_p

where:
R_p = Portfolio return (annualized)
R_f = Risk-free rate (6.5%)
σ_p = Portfolio volatility (annualized)

Annualization:
Annual return = (1 + daily_return)^252 - 1
Annual volatility = daily_volatility × √252
```

**Sortino Ratio:**
```
Sortino = (R_p - R_f) / σ_downside

where:
σ_downside = √(Σ min(0, R_i - R_f)² / n)
```

**Calmar Ratio:**
```
Calmar = Annual Return / |Maximum Drawdown|
```

### B.2 Drawdown Calculation

**Drawdown:**
```
DD_t = (Peak_t - Value_t) / Peak_t

where:
Peak_t = max(Value₁, Value₂, ..., Value_t)
```

**Maximum Drawdown:**
```
MDD = max_t (DD_t)
```

### B.3 Win Rate and Expectancy

**Win Rate:**
```
WR = Number of Winning Trades / Total Trades
```

**Average Win/Loss:**
```
Avg Win = Σ(Winning Trades) / Number of Wins
Avg Loss = Σ(Losing Trades) / Number of Losses
```

**Expectancy:**
```
E = (WR × Avg Win) - ((1 - WR) × |Avg Loss|)
```

**Profit Factor:**
```
PF = Gross Profit / |Gross Loss|
```

## C. Statistical Tests

### C.1 Z-Score for Outlier Detection

**Formula:**
```
z = (x - μ) / σ

where:
x = Observation
μ = Mean
σ = Standard deviation

Outlier if |z| > 3
```

### C.2 T-Test for Mean Comparison

**Independent Samples T-Test:**
```
t = (μ₁ - μ₂) / √(s₁²/n₁ + s₂²/n₂)

where:
μ₁, μ₂ = Sample means
s₁², s₂² = Sample variances
n₁, n₂ = Sample sizes

df = n₁ + n₂ - 2
```

### C.3 Chi-Square Test

**Test Statistic:**
```
χ² = Σ (O_i - E_i)² / E_i

where:
O_i = Observed frequency
E_i = Expected frequency

df = (rows - 1) × (columns - 1)
```

## D. Implementation Details

### D.1 Data Structures

**Spot Data:**
```python
columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
dtype = {'timestamp': datetime64, 'open': float64, ...}
```

**Futures Data:**
```python
columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi']
```

**Options Data:**
```python
columns = ['timestamp', 'strike', 'option_type', 'ltp', 'iv', 'oi', 'volume']
option_type = ['CE', 'PE']  # Call European, Put European
```

### D.2 Feature Naming Convention

**Technical Indicators:**
```
ema_{period}
rsi_{period}
bb_upper_{period}_{std}
bb_lower_{period}_{std}
```

**Options Greeks:**
```
{greek}_{strike_offset}_{option_type}
Examples:
- delta_atm_call
- gamma_plus1_put
- theta_minus2_call
```

**Derived Features:**
```
iv_avg
iv_skew
pcr_oi
pcr_volume
futures_basis
delta_skew
gamma_exposure
```

### D.3 Model Serialization

**Pickle (HMM, XGBoost, Scaler):**
```python
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

**HDF5 (LSTM):**
```python
model.save('model.h5')
```

### D.4 Configuration Management

**Centralized Config:**
```python
# src/config.py
START_DATE = datetime.now() - timedelta(days=365)
INITIAL_CAPITAL = 1000000
STOP_LOSS_PCT = 0.02
```

**Usage:**
```python
from src.config import INITIAL_CAPITAL, STOP_LOSS_PCT
```

## E. Code Quality Standards

### E.1 Docstring Format

```python
def calculate_greeks(spot, strike, time_to_expiry, iv, option_type):
    """
    Calculate Black-Scholes Greeks for an option.
    
    Parameters
    ----------
    spot : float
        Current spot price
    strike : float
        Option strike price
    time_to_expiry : float
        Time to expiry in years
    iv : float
        Implied volatility (decimal, e.g., 0.20 for 20%)
    option_type : str
        'call' or 'put'
    
    Returns
    -------
    dict
        Dictionary with keys: delta, gamma, theta, vega, rho
    
    Examples
    --------
    >>> greeks = calculate_greeks(18000, 18000, 0.0833, 0.15, 'call')
    >>> greeks['delta']
    0.5234
    """
```

### E.2 Error Handling

```python
try:
    result = risky_operation()
except SpecificException as e:
    logger.error(f"Operation failed: {e}")
    # Handle gracefully
    result = default_value
finally:
    # Cleanup
    pass
```

### E.3 Logging Standards

```python
import logging
logger = logging.getLogger(__name__)

logger.info("Starting process")
logger.warning("Potential issue detected")
logger.error("Operation failed")
logger.debug("Detailed debug information")
```

## F. Performance Optimization

### F.1 Vectorization

**Avoid:**
```python
for i in range(len(df)):
    df.loc[i, 'result'] = df.loc[i, 'a'] + df.loc[i, 'b']
```

**Prefer:**
```python
df['result'] = df['a'] + df['b']
```

### F.2 Memory Management

**Data Types:**
```python
# Use appropriate dtypes
df['price'] = df['price'].astype('float32')  # Instead of float64
df['volume'] = df['volume'].astype('int32')  # Instead of int64
```

**Chunking:**
```python
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    process(chunk)
```

### F.3 Parallel Processing

```python
from joblib import Parallel, delayed

results = Parallel(n_jobs=-1)(
    delayed(process_function)(item) for item in items
)
```

## G. Testing Strategy

### G.1 Unit Tests

```python
import unittest

class TestGreeksCalculation(unittest.TestCase):
    def test_atm_call_delta(self):
        greeks = calculate_greeks(18000, 18000, 0.0833, 0.15, 'call')
        self.assertAlmostEqual(greeks['delta'], 0.5, places=1)
    
    def test_put_call_parity(self):
        # Test put-call parity relationship
        pass
```

### G.2 Integration Tests

```python
def test_complete_pipeline():
    # Test end-to-end pipeline
    fetch_data()
    clean_data()
    create_features()
    # Assert expected outputs exist
    assert os.path.exists('data/features/nifty_features_5min.csv')
```

### G.3 Validation Tests

```python
def test_data_quality():
    df = pd.read_csv('data/processed/nifty_merged_5min.csv')
    # No missing values in critical columns
    assert df[['close', 'volume']].isna().sum().sum() == 0
    # Prices are positive
    assert (df['close'] > 0).all()
    # Timestamps are sorted
    assert df['timestamp'].is_monotonic_increasing
```

## H. Deployment Considerations

### H.1 Environment Management

```bash
# requirements.txt with versions
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.2.2
xgboost==2.0.3
tensorflow==2.13.0
```

### H.2 Configuration for Production

```python
# config/production.py
LIVE_DATA_SOURCE = 'NSE_API'
ENABLE_LOGGING = True
LOG_LEVEL = 'INFO'
ALERT_EMAIL = 'trader@example.com'
```

### H.3 Monitoring

```python
# Track key metrics
metrics = {
    'timestamp': datetime.now(),
    'portfolio_value': current_value,
    'open_positions': len(positions),
    'daily_pnl': calculate_pnl(),
    'sharpe_ratio': calculate_sharpe()
}
log_metrics(metrics)
```

## I. References

### I.1 Academic Papers
1. Baum, L. E., & Petrie, T. (1966). "Statistical Inference for Probabilistic Functions of Finite State Markov Chains"
2. Black, F., & Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities"
3. Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System"
4. Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory"

### I.2 Books
1. Hull, J. C. (2017). "Options, Futures, and Other Derivatives" (10th ed.)
2. López de Prado, M. (2018). "Advances in Financial Machine Learning"
3. Chan, E. P. (2013). "Algorithmic Trading: Winning Strategies and Their Rationale"

### I.3 Online Resources
1. NSE India: https://www.nseindia.com/
2. QuantLib: https://www.quantlib.org/
3. XGBoost Documentation: https://xgboost.readthedocs.io/
4. TensorFlow Documentation: https://www.tensorflow.org/

## J. Glossary

**ATM (At-The-Money)**: Strike price equal to current spot price

**Basis**: Difference between futures and spot price

**Delta**: Rate of change of option price w.r.t. underlying

**Gamma**: Rate of change of delta

**Greeks**: Sensitivity measures for options (Delta, Gamma, Theta, Vega, Rho)

**HMM**: Hidden Markov Model - statistical model for sequential data

**IV (Implied Volatility)**: Market's expectation of future volatility

**OI (Open Interest)**: Total number of outstanding contracts

**PCR (Put-Call Ratio)**: Ratio of put to call volume or open interest

**Regime**: Market state (Uptrend, Sideways, Downtrend)

**Sharpe Ratio**: Risk-adjusted return metric

**Theta**: Time decay of option value

**Vega**: Sensitivity to volatility changes

**Viterbi Algorithm**: Dynamic programming algorithm for HMM inference
