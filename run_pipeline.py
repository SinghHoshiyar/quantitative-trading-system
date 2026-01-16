"""
Main pipeline script to run the complete quantitative trading system.
"""
import sys
import logging
from datetime import datetime

# Import all modules
from src.data_acquisition.fetch_data import main as fetch_data
from src.data_acquisition.clean_data import main as clean_data
from src.feature_engineering.create_features import main as create_features
from src.regime_detection.hmm_regimes import main as detect_regimes
from src.strategy.ema_strategy import main as run_strategy
from src.ml_models.train_models import main as train_models
from src.analysis.outlier_analysis import main as analyze_outliers

from src.utils import setup_logger


logger = setup_logger(__name__)


def run_complete_pipeline():
    """Run the complete quantitative trading pipeline."""
    
    start_time = datetime.now()
    
    logger.info("="*80)
    logger.info("QUANTITATIVE TRADING SYSTEM - COMPLETE PIPELINE")
    logger.info("="*80)
    
    try:
        # Step 1: Data Acquisition
        logger.info("\n" + "="*80)
        logger.info("STEP 1: DATA ACQUISITION")
        logger.info("="*80)
        fetch_data()
        
        # Step 2: Data Cleaning
        logger.info("\n" + "="*80)
        logger.info("STEP 2: DATA CLEANING & MERGING")
        logger.info("="*80)
        clean_data()
        
        # Step 3: Feature Engineering
        logger.info("\n" + "="*80)
        logger.info("STEP 3: FEATURE ENGINEERING")
        logger.info("="*80)
        create_features()
        
        # Step 4: Regime Detection
        logger.info("\n" + "="*80)
        logger.info("STEP 4: REGIME DETECTION (HMM)")
        logger.info("="*80)
        detect_regimes()
        
        # Step 5: Baseline Strategy
        logger.info("\n" + "="*80)
        logger.info("STEP 5: EMA STRATEGY BACKTEST")
        logger.info("="*80)
        run_strategy()
        
        # Step 6: ML Model Training
        logger.info("\n" + "="*80)
        logger.info("STEP 6: MACHINE LEARNING MODEL TRAINING")
        logger.info("="*80)
        train_models()
        
        # Step 7: Outlier Analysis
        logger.info("\n" + "="*80)
        logger.info("STEP 7: HIGH-PERFORMANCE TRADE ANALYSIS")
        logger.info("="*80)
        analyze_outliers()
        
        # Pipeline completed
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"Total execution time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        logger.info(f"Results saved in: results/")
        logger.info(f"Models saved in: models/")
        logger.info("="*80)
        
        return True
        
    except Exception as e:
        logger.error(f"\n{'='*80}")
        logger.error(f"PIPELINE FAILED: {str(e)}")
        logger.error(f"{'='*80}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_complete_pipeline()
    sys.exit(0 if success else 1)
