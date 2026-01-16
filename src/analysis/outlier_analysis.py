"""
High-performance trade analysis and outlier detection.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import sys
sys.path.append('.')

from src.config import *
from src.utils import setup_logger, load_dataframe, detect_outliers


logger = setup_logger(__name__)


class OutlierAnalyzer:
    """Analyze high-performance trades."""
    
    def __init__(self, threshold: float = OUTLIER_THRESHOLD):
        self.threshold = threshold
    
    def identify_outliers(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Identify outlier trades using z-score."""
        logger.info(f"Identifying outlier trades (threshold: {self.threshold} sigma)")
        
        if 'return' not in trades_df.columns or trades_df.empty:
            logger.warning("No trade returns found")
            return pd.DataFrame()
        
        # Calculate z-scores
        returns = trades_df['return']
        z_scores = np.abs((returns - returns.mean()) / returns.std())
        
        # Identify outliers
        outliers = trades_df[z_scores > self.threshold].copy()
        outliers['z_score'] = z_scores[z_scores > self.threshold]
        
        logger.info(f"Found {len(outliers)} outlier trades ({len(outliers)/len(trades_df)*100:.1f}%)")
        
        # Separate positive and negative outliers
        positive_outliers = outliers[outliers['return'] > 0]
        negative_outliers = outliers[outliers['return'] < 0]
        
        logger.info(f"Positive outliers: {len(positive_outliers)}, Negative outliers: {len(negative_outliers)}")
        
        return outliers
    
    def analyze_outlier_patterns(self, trades_df: pd.DataFrame, outliers_df: pd.DataFrame,
                                 features_df: pd.DataFrame) -> dict:
        """Analyze patterns in outlier trades."""
        logger.info("Analyzing outlier patterns")
        
        if outliers_df.empty:
            return {}
        
        # Merge with features
        outliers_with_features = self._merge_trade_features(outliers_df, features_df)
        normal_trades = trades_df[~trades_df.index.isin(outliers_df.index)]
        normal_with_features = self._merge_trade_features(normal_trades, features_df)
        
        analysis = {}
        
        # 1. Regime analysis
        if 'regime' in outliers_with_features.columns:
            outlier_regimes = outliers_with_features['regime'].value_counts()
            normal_regimes = normal_with_features['regime'].value_counts()
            
            analysis['regime_distribution'] = {
                'outliers': outlier_regimes.to_dict(),
                'normal': normal_regimes.to_dict()
            }
            
            logger.info(f"Outlier regime distribution: {outlier_regimes.to_dict()}")
        
        # 2. Time of day analysis
        outliers_with_features['hour'] = pd.to_datetime(outliers_with_features['entry_time']).dt.hour
        normal_with_features['hour'] = pd.to_datetime(normal_with_features['entry_time']).dt.hour
        
        outlier_hours = outliers_with_features['hour'].value_counts()
        normal_hours = normal_with_features['hour'].value_counts()
        
        analysis['time_distribution'] = {
            'outliers': outlier_hours.to_dict(),
            'normal': normal_hours.to_dict()
        }
        
        # 3. IV environment analysis
        if 'iv_avg' in outliers_with_features.columns:
            analysis['iv_stats'] = {
                'outliers': {
                    'mean': outliers_with_features['iv_avg'].mean(),
                    'std': outliers_with_features['iv_avg'].std()
                },
                'normal': {
                    'mean': normal_with_features['iv_avg'].mean(),
                    'std': normal_with_features['iv_avg'].std()
                }
            }
            
            logger.info(f"Outlier IV: {analysis['iv_stats']['outliers']['mean']:.4f}, "
                       f"Normal IV: {analysis['iv_stats']['normal']['mean']:.4f}")
        
        # 4. Duration analysis
        if 'duration' in outliers_df.columns:
            analysis['duration_stats'] = {
                'outliers': {
                    'mean': outliers_df['duration'].mean(),
                    'median': outliers_df['duration'].median()
                },
                'normal': {
                    'mean': normal_trades['duration'].mean(),
                    'median': normal_trades['duration'].median()
                }
            }
        
        # 5. Statistical tests
        if len(outliers_with_features) > 5 and len(normal_with_features) > 5:
            # T-test for IV difference
            if 'iv_avg' in outliers_with_features.columns:
                t_stat, p_value = stats.ttest_ind(
                    outliers_with_features['iv_avg'].dropna(),
                    normal_with_features['iv_avg'].dropna()
                )
                analysis['iv_ttest'] = {'t_statistic': t_stat, 'p_value': p_value}
                logger.info(f"IV t-test: t={t_stat:.4f}, p={p_value:.4f}")
        
        return analysis
    
    def _merge_trade_features(self, trades_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
        """Merge trade data with features at entry time."""
        trades_with_features = trades_df.copy()
        
        # Convert timestamps to datetime
        trades_with_features['entry_time'] = pd.to_datetime(trades_with_features['entry_time'])
        features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])
        
        # Convert timestamps
        trades_with_features['entry_time'] = pd.to_datetime(trades_with_features['entry_time'])
        features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])
        
        # Merge on nearest timestamp
        merged = pd.merge_asof(
            trades_with_features.sort_values('entry_time'),
            features_df.sort_values('timestamp'),
            left_on='entry_time',
            right_on='timestamp',
            direction='nearest'
        )
        
        return merged
    
    def visualize_outliers(self, trades_df: pd.DataFrame, outliers_df: pd.DataFrame, save_path: str):
        """Create visualizations for outlier analysis."""
        logger.info("Creating outlier visualizations")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Return distribution with outliers highlighted
        axes[0, 0].hist(trades_df['return'], bins=50, alpha=0.7, label='All Trades')
        axes[0, 0].hist(outliers_df['return'], bins=20, alpha=0.7, label='Outliers', color='red')
        axes[0, 0].axvline(trades_df['return'].mean(), color='blue', linestyle='--', label='Mean')
        axes[0, 0].axvline(trades_df['return'].mean() + self.threshold * trades_df['return'].std(),
                          color='red', linestyle='--', label=f'{self.threshold}σ')
        axes[0, 0].axvline(trades_df['return'].mean() - self.threshold * trades_df['return'].std(),
                          color='red', linestyle='--')
        axes[0, 0].set_xlabel('Return')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Trade Return Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Outlier returns over time
        outliers_sorted = outliers_df.sort_values('entry_time')
        axes[0, 1].scatter(range(len(outliers_sorted)), outliers_sorted['return'],
                          c=outliers_sorted['return'], cmap='RdYlGn', s=100)
        axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel('Outlier Trade Index')
        axes[0, 1].set_ylabel('Return')
        axes[0, 1].set_title('Outlier Trade Returns')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Duration comparison
        if 'duration' in trades_df.columns and 'duration' in outliers_df.columns:
            normal_trades = trades_df[~trades_df.index.isin(outliers_df.index)]
            
            data_to_plot = [normal_trades['duration'].dropna(), outliers_df['duration'].dropna()]
            axes[1, 0].boxplot(data_to_plot, labels=['Normal', 'Outliers'])
            axes[1, 0].set_ylabel('Duration (hours)')
            axes[1, 0].set_title('Trade Duration Comparison')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Return vs Duration scatter
        if 'duration' in trades_df.columns:
            axes[1, 1].scatter(trades_df['duration'], trades_df['return'],
                             alpha=0.5, label='Normal', s=20)
            axes[1, 1].scatter(outliers_df['duration'], outliers_df['return'],
                             alpha=0.8, label='Outliers', s=100, color='red')
            axes[1, 1].set_xlabel('Duration (hours)')
            axes[1, 1].set_ylabel('Return')
            axes[1, 1].set_title('Return vs Duration')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved outlier visualization to {save_path}")
    
    def generate_insights(self, analysis: dict) -> list:
        """Generate actionable insights from outlier analysis."""
        insights = []
        
        # Regime insights
        if 'regime_distribution' in analysis:
            outlier_regimes = analysis['regime_distribution']['outliers']
            if outlier_regimes:
                dominant_regime = max(outlier_regimes, key=outlier_regimes.get)
                insights.append(
                    f"Most high-performance trades occur in regime {REGIME_LABELS.get(dominant_regime, dominant_regime)} "
                    f"({outlier_regimes[dominant_regime]} trades)"
                )
        
        # Time insights
        if 'time_distribution' in analysis:
            outlier_hours = analysis['time_distribution']['outliers']
            if outlier_hours:
                best_hour = max(outlier_hours, key=outlier_hours.get)
                insights.append(
                    f"Peak performance hour: {best_hour}:00 ({outlier_hours[best_hour]} outlier trades)"
                )
        
        # IV insights
        if 'iv_stats' in analysis and 'iv_ttest' in analysis:
            outlier_iv = analysis['iv_stats']['outliers']['mean']
            normal_iv = analysis['iv_stats']['normal']['mean']
            p_value = analysis['iv_ttest']['p_value']
            
            if p_value < 0.05:
                if outlier_iv > normal_iv:
                    insights.append(
                        f"High-performance trades occur in higher IV environments "
                        f"(Outlier IV: {outlier_iv:.2%} vs Normal: {normal_iv:.2%}, p={p_value:.4f})"
                    )
                else:
                    insights.append(
                        f"High-performance trades occur in lower IV environments "
                        f"(Outlier IV: {outlier_iv:.2%} vs Normal: {normal_iv:.2%}, p={p_value:.4f})"
                    )
        
        # Duration insights
        if 'duration_stats' in analysis:
            outlier_duration = analysis['duration_stats']['outliers']['median']
            normal_duration = analysis['duration_stats']['normal']['median']
            
            if outlier_duration < normal_duration:
                insights.append(
                    f"High-performance trades are typically shorter "
                    f"(Median: {outlier_duration:.1f}h vs {normal_duration:.1f}h)"
                )
            else:
                insights.append(
                    f"High-performance trades require more time to develop "
                    f"(Median: {outlier_duration:.1f}h vs {normal_duration:.1f}h)"
                )
        
        return insights


def main():
    """Main function for outlier analysis."""
    logger.info("Starting outlier analysis")
    
    # Load trades and features
    try:
        trades_df = load_dataframe(f'{RESULTS_DIR}/ema_strategy_trades.csv', logger)
        features_df = load_dataframe(REGIME_FILE, logger)
    except:
        logger.error("Could not load trade data. Run strategy backtest first.")
        return
    
    # Initialize analyzer
    analyzer = OutlierAnalyzer()
    
    # Identify outliers
    outliers_df = analyzer.identify_outliers(trades_df)
    
    if not outliers_df.empty:
        # Analyze patterns
        analysis = analyzer.analyze_outlier_patterns(trades_df, outliers_df, features_df)
        
        # Visualize
        analyzer.visualize_outliers(trades_df, outliers_df,
                                    f'{RESULTS_DIR}/outlier_analysis.png')
        
        # Generate insights
        insights = analyzer.generate_insights(analysis)
        
        logger.info("\n" + "="*60)
        logger.info("KEY INSIGHTS FROM HIGH-PERFORMANCE TRADES")
        logger.info("="*60)
        for i, insight in enumerate(insights, 1):
            logger.info(f"{i}. {insight}")
        logger.info("="*60)
        
        # Save outliers
        outliers_df.to_csv(f'{RESULTS_DIR}/outlier_trades.csv', index=False)
        logger.info(f"Saved outlier trades to {RESULTS_DIR}/outlier_trades.csv")
    
    logger.info("Outlier analysis completed successfully")


if __name__ == "__main__":
    main()
