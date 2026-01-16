"""
EMA crossover strategy with regime filtering.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('.')

from src.config import *
from src.utils import (setup_logger, load_dataframe, save_dataframe,
                       calculate_sharpe_ratio, calculate_sortino_ratio,
                       calculate_max_drawdown, calculate_calmar_ratio,
                       calculate_win_rate, print_performance_summary)


logger = setup_logger(__name__)


class EMAStrategy:
    """EMA crossover strategy with regime filtering."""
    
    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.position_size = POSITION_SIZE
        self.stop_loss = STOP_LOSS_PCT
        self.take_profit = TAKE_PROFIT_PCT
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on EMA crossover and regime."""
        logger.info("Generating trading signals")
        
        df = df.copy()
        
        # EMA crossover signals
        df['ema_signal'] = 0
        df.loc[df[f'ema_{EMA_SHORT}'] > df[f'ema_{EMA_LONG}'], 'ema_signal'] = 1  # Bullish
        df.loc[df[f'ema_{EMA_SHORT}'] < df[f'ema_{EMA_LONG}'], 'ema_signal'] = -1  # Bearish
        
        # Regime filter
        df['regime_filter'] = 0
        df.loc[df['regime'].isin(ALLOWED_LONG_REGIMES), 'regime_filter'] = 1
        df.loc[df['regime'].isin(ALLOWED_SHORT_REGIMES), 'regime_filter'] = -1
        
        # Combined signal (only trade when regime agrees)
        df['signal'] = 0
        df.loc[(df['ema_signal'] == 1) & (df['regime_filter'] == 1), 'signal'] = 1  # Long
        df.loc[(df['ema_signal'] == -1) & (df['regime_filter'] == -1), 'signal'] = -1  # Short
        
        # Generate entry/exit signals
        df['position'] = df['signal'].shift(1).fillna(0)
        df['entry'] = (df['position'] != df['position'].shift(1)) & (df['position'] != 0)
        df['exit'] = (df['position'] != df['position'].shift(1)) & (df['position'].shift(1) != 0)
        
        logger.info(f"Generated {df['entry'].sum()} entry signals")
        
        return df
    
    def backtest(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run backtest with position sizing and risk management."""
        logger.info("Running backtest")
        
        df = df.copy()
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Initialize tracking columns
        df['capital'] = float(self.initial_capital)
        df['position_size'] = 0.0
        df['entry_price'] = 0.0
        df['pnl'] = 0.0
        df['trade_return'] = 0.0
        
        capital = self.initial_capital
        position = 0
        entry_price = 0
        position_qty = 0
        
        trades = []
        
        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            # Check for entry
            if row['entry'] and position == 0:
                position = row['position']
                entry_price = row['close']
                position_qty = (capital * self.position_size) / entry_price
                
                trades.append({
                    'entry_time': row['timestamp'],
                    'entry_price': entry_price,
                    'position': position,
                    'quantity': position_qty
                })
            
            # Check for exit (signal change or stop loss/take profit)
            elif position != 0:
                exit_signal = False
                exit_reason = ''
                
                # Signal-based exit
                if row['exit']:
                    exit_signal = True
                    exit_reason = 'signal'
                
                # Stop loss
                elif position == 1 and (row['close'] - entry_price) / entry_price <= -self.stop_loss:
                    exit_signal = True
                    exit_reason = 'stop_loss'
                
                elif position == -1 and (entry_price - row['close']) / entry_price <= -self.stop_loss:
                    exit_signal = True
                    exit_reason = 'stop_loss'
                
                # Take profit
                elif position == 1 and (row['close'] - entry_price) / entry_price >= self.take_profit:
                    exit_signal = True
                    exit_reason = 'take_profit'
                
                elif position == -1 and (entry_price - row['close']) / entry_price >= self.take_profit:
                    exit_signal = True
                    exit_reason = 'take_profit'
                
                if exit_signal:
                    # Calculate PnL
                    if position == 1:
                        pnl = (row['close'] - entry_price) * position_qty
                    else:
                        pnl = (entry_price - row['close']) * position_qty
                    
                    trade_return = pnl / capital
                    capital += pnl
                    
                    df.loc[i, 'pnl'] = pnl
                    df.loc[i, 'trade_return'] = trade_return
                    
                    # Update last trade
                    if trades:
                        trades[-1].update({
                            'exit_time': row['timestamp'],
                            'exit_price': row['close'],
                            'pnl': pnl,
                            'return': trade_return,
                            'exit_reason': exit_reason,
                            'duration': (row['timestamp'] - trades[-1]['entry_time']).total_seconds() / 3600
                        })
                    
                    # Reset position
                    position = 0
                    entry_price = 0
                    position_qty = 0
            
            df.loc[i, 'capital'] = capital
        
        self.trades_df = pd.DataFrame(trades)
        
        logger.info(f"Backtest completed - Total trades: {len(trades)}")
        logger.info(f"Final capital: ₹{capital:,.2f}")
        
        return df
    
    def calculate_metrics(self, df: pd.DataFrame) -> dict:
        """Calculate performance metrics."""
        logger.info("Calculating performance metrics")
        
        # Filter completed trades
        trade_returns = df[df['trade_return'] != 0]['trade_return']
        
        if len(trade_returns) == 0:
            logger.warning("No completed trades found")
            return {}
        
        metrics = {
            'total_return': (df['capital'].iloc[-1] / self.initial_capital) - 1,
            'total_trades': len(trade_returns),
            'winning_trades': (trade_returns > 0).sum(),
            'losing_trades': (trade_returns < 0).sum(),
            'win_rate': calculate_win_rate(trade_returns),
            'avg_return': trade_returns.mean(),
            'avg_win': trade_returns[trade_returns > 0].mean() if (trade_returns > 0).any() else 0,
            'avg_loss': trade_returns[trade_returns < 0].mean() if (trade_returns < 0).any() else 0,
            'sharpe_ratio': calculate_sharpe_ratio(trade_returns),
            'sortino_ratio': calculate_sortino_ratio(trade_returns),
            'max_drawdown': calculate_max_drawdown(trade_returns),
            'calmar_ratio': calculate_calmar_ratio(trade_returns)
        }
        
        if hasattr(self, 'trades_df') and not self.trades_df.empty:
            metrics['avg_trade_duration_hours'] = self.trades_df['duration'].mean()
        
        return metrics
    
    def plot_results(self, df: pd.DataFrame, save_path: str):
        """Plot backtest results."""
        logger.info("Creating backtest visualization")
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot 1: Price and signals
        axes[0].plot(df['timestamp'], df['close'], label='Close Price', linewidth=0.5)
        axes[0].plot(df['timestamp'], df[f'ema_{EMA_SHORT}'], label=f'EMA {EMA_SHORT}', alpha=0.7)
        axes[0].plot(df['timestamp'], df[f'ema_{EMA_LONG}'], label=f'EMA {EMA_LONG}', alpha=0.7)
        
        # Mark entries and exits
        entries = df[df['entry']]
        exits = df[df['exit']]
        
        axes[0].scatter(entries['timestamp'], entries['close'], 
                       c='green', marker='^', s=100, label='Entry', zorder=5)
        axes[0].scatter(exits['timestamp'], exits['close'],
                       c='red', marker='v', s=100, label='Exit', zorder=5)
        
        axes[0].set_title('Trading Signals')
        axes[0].set_ylabel('Price')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Equity curve
        axes[1].plot(df['timestamp'], df['capital'], label='Portfolio Value', linewidth=1)
        axes[1].axhline(y=self.initial_capital, color='r', linestyle='--', label='Initial Capital')
        axes[1].set_title('Equity Curve')
        axes[1].set_ylabel('Capital (₹)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Drawdown
        cumulative_returns = (df['capital'] / self.initial_capital) - 1
        running_max = (1 + cumulative_returns).cummax()
        drawdown = ((1 + cumulative_returns) / running_max) - 1
        
        axes[2].fill_between(df['timestamp'], drawdown, 0, alpha=0.3, color='red')
        axes[2].set_title('Drawdown')
        axes[2].set_ylabel('Drawdown')
        axes[2].set_xlabel('Date')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved backtest visualization to {save_path}")


def main():
    """Main function to run EMA strategy."""
    logger.info("Starting EMA strategy backtest")
    
    # Load data with regimes
    df = load_dataframe(REGIME_FILE, logger)
    
    # Initialize strategy
    strategy = EMAStrategy()
    
    # Generate signals
    df = strategy.generate_signals(df)
    
    # Run backtest
    df = strategy.backtest(df)
    
    # Calculate metrics
    metrics = strategy.calculate_metrics(df)
    
    # Print results
    print_performance_summary(metrics, logger)
    
    # Plot results
    strategy.plot_results(df, f'{RESULTS_DIR}/ema_strategy_results.png')
    
    # Save results
    save_dataframe(df, f'{RESULTS_DIR}/ema_strategy_backtest.csv', logger)
    if hasattr(strategy, 'trades_df'):
        save_dataframe(strategy.trades_df, f'{RESULTS_DIR}/ema_strategy_trades.csv', logger)
    
    logger.info("EMA strategy backtest completed successfully")


if __name__ == "__main__":
    main()
