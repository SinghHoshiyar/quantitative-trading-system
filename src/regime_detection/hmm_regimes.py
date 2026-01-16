"""
Hidden Markov Model for market regime detection.
"""
import pandas as pd
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('.')

from src.config import *
from src.utils import setup_logger, load_dataframe, save_dataframe, save_model


logger = setup_logger(__name__)


class RegimeDetector:
    """Detect market regimes using HMM."""
    
    def __init__(self, n_regimes: int = N_REGIMES):
        self.n_regimes = n_regimes
        self.model = None
        self.feature_columns = REGIME_FEATURES
    
    def fit(self, df: pd.DataFrame) -> 'RegimeDetector':
        """Fit HMM model on features."""
        logger.info(f"Fitting HMM with {self.n_regimes} regimes")
        
        # Select features for regime detection
        available_features = [f for f in self.feature_columns if f in df.columns]
        logger.info(f"Using features: {available_features}")
        
        X = df[available_features].values
        
        # Normalize features
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
        
        # Initialize and fit HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=HMM_COVARIANCE_TYPE,
            n_iter=HMM_N_ITER,
            random_state=RANDOM_STATE
        )
        
        self.model.fit(X)
        
        logger.info("HMM fitting completed")
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict regimes."""
        available_features = [f for f in self.feature_columns if f in df.columns]
        X = df[available_features].values
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
        
        regimes = self.model.predict(X)
        
        # Map regimes to meaningful labels (0=Down, 1=Sideways, 2=Up)
        regimes = self._map_regimes(df, regimes)
        
        return regimes
    
    def _map_regimes(self, df: pd.DataFrame, regimes: np.ndarray) -> np.ndarray:
        """Map regime numbers to meaningful labels based on returns."""
        # Calculate average return for each regime
        df_temp = df.copy()
        df_temp['regime'] = regimes
        df_temp['returns'] = df_temp['close'].pct_change()
        
        regime_returns = df_temp.groupby('regime')['returns'].mean().sort_values()
        
        # Create mapping: lowest return = 0 (down), highest = 2 (up)
        mapping = {old: new for new, old in enumerate(regime_returns.index)}
        
        mapped_regimes = np.array([mapping[r] for r in regimes])
        
        logger.info(f"Regime mapping: {mapping}")
        logger.info(f"Regime returns: {regime_returns.to_dict()}")
        
        return mapped_regimes
    
    def visualize_regimes(self, df: pd.DataFrame, regimes: np.ndarray, save_path: str):
        """Visualize regimes over time."""
        logger.info("Creating regime visualization")
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        
        # Plot 1: Price with regime colors
        for regime in range(self.n_regimes):
            mask = regimes == regime
            axes[0].scatter(df.loc[mask, 'timestamp'], df.loc[mask, 'close'],
                          c=f'C{regime}', label=REGIME_LABELS[regime], alpha=0.5, s=1)
        axes[0].set_title('NIFTY Price with Market Regimes')
        axes[0].set_ylabel('Price')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Regime timeline
        axes[1].plot(df['timestamp'], regimes, linewidth=0.5)
        axes[1].set_title('Regime Timeline')
        axes[1].set_ylabel('Regime')
        axes[1].set_yticks([0, 1, 2])
        axes[1].set_yticklabels(['Downtrend', 'Sideways', 'Uptrend'])
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Regime distribution
        regime_counts = pd.Series(regimes).value_counts().sort_index()
        axes[2].bar([REGIME_LABELS[i] for i in regime_counts.index], regime_counts.values)
        axes[2].set_title('Regime Distribution')
        axes[2].set_ylabel('Count')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved regime visualization to {save_path}")
    
    def analyze_regimes(self, df: pd.DataFrame, regimes: np.ndarray):
        """Analyze regime characteristics."""
        logger.info("Analyzing regime characteristics")
        
        df_analysis = df.copy()
        df_analysis['regime'] = regimes
        df_analysis['returns'] = df_analysis['close'].pct_change()
        
        # Statistics by regime
        stats = df_analysis.groupby('regime').agg({
            'returns': ['mean', 'std', 'count'],
            'volume': 'mean',
            'close': ['min', 'max']
        })
        
        logger.info(f"\nRegime Statistics:\n{stats}")
        
        # Transition matrix
        transitions = self._calculate_transition_matrix(regimes)
        logger.info(f"\nTransition Matrix:\n{transitions}")
        
        return stats, transitions
    
    def _calculate_transition_matrix(self, regimes: np.ndarray) -> pd.DataFrame:
        """Calculate regime transition probabilities."""
        n = self.n_regimes
        transitions = np.zeros((n, n))
        
        for i in range(len(regimes) - 1):
            transitions[regimes[i], regimes[i+1]] += 1
        
        # Normalize to probabilities
        row_sums = transitions.sum(axis=1, keepdims=True)
        transitions = transitions / (row_sums + 1e-10)
        
        df_transitions = pd.DataFrame(
            transitions,
            index=[REGIME_LABELS[i] for i in range(n)],
            columns=[REGIME_LABELS[i] for i in range(n)]
        )
        
        return df_transitions


def main():
    """Main function for regime detection."""
    logger.info("Starting regime detection process")
    
    # Load features
    df = load_dataframe(FEATURES_FILE, logger)
    
    # Initialize and fit detector
    detector = RegimeDetector()
    detector.fit(df)
    
    # Predict regimes
    regimes = detector.predict(df)
    df['regime'] = regimes
    
    # Save model
    save_model(detector.model, HMM_MODEL_FILE, logger)
    
    # Save data with regimes
    save_dataframe(df, REGIME_FILE, logger)
    
    # Visualize regimes
    detector.visualize_regimes(df, regimes, f'{RESULTS_DIR}/regime_visualization.png')
    
    # Analyze regimes
    detector.analyze_regimes(df, regimes)
    
    logger.info("Regime detection completed successfully")


if __name__ == "__main__":
    main()
