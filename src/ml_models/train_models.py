"""
Train machine learning models for trade prediction.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('.')

from src.config import *
from src.utils import setup_logger, load_dataframe, save_model
import joblib


logger = setup_logger(__name__)


class MLTrainer:
    """Train ML models for trade prediction."""
    
    def __init__(self):
        self.xgb_model = None
        self.lstm_model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
    
    def prepare_data(self, df: pd.DataFrame) -> tuple:
        """Prepare data for ML training."""
        logger.info("Preparing data for ML training")
        
        df = df.copy()
        
        # Create target: will next trade be profitable?
        # Look ahead to find next trade outcome
        df['future_return'] = df['close'].shift(-20) / df['close'] - 1
        df['target'] = (df['future_return'] > 0).astype(int)
        
        # Select features
        self.feature_columns = [col for col in ML_FEATURE_COLUMNS if col in df.columns]
        logger.info(f"Using {len(self.feature_columns)} features")
        
        # Remove rows with NaN
        df = df.dropna(subset=self.feature_columns + ['target'])
        
        X = df[self.feature_columns].values
        y = df['target'].values
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(1-TRAIN_SIZE), random_state=RANDOM_STATE, shuffle=False
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=TEST_SIZE/(TEST_SIZE+VALIDATION_SIZE),
            random_state=RANDOM_STATE, shuffle=False
        )
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        
        logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        logger.info(f"Target distribution - Train: {np.mean(y_train):.2%}, Test: {np.mean(y_test):.2%}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_xgboost(self, X_train, X_val, y_train, y_val):
        """Train XGBoost model."""
        logger.info("Training XGBoost model")
        
        # Update params for newer XGBoost API
        xgb_params = XGBOOST_PARAMS.copy()
        xgb_params['early_stopping_rounds'] = 20
        xgb_params['eval_metric'] = 'logloss'
        
        self.xgb_model = xgb.XGBClassifier(**xgb_params)
        
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Evaluate
        train_pred = self.xgb_model.predict(X_train)
        val_pred = self.xgb_model.predict(X_val)
        
        train_acc = (train_pred == y_train).mean()
        val_acc = (val_pred == y_val).mean()
        
        logger.info(f"XGBoost - Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}")
        
        return self.xgb_model
    
    def train_lstm(self, X_train, X_val, y_train, y_val):
        """Train LSTM model."""
        logger.info("Training LSTM model")
        
        # Reshape for LSTM (samples, timesteps, features)
        seq_len = LSTM_PARAMS['sequence_length']
        
        X_train_seq, y_train_seq = self._create_sequences(X_train, y_train, seq_len)
        X_val_seq, y_val_seq = self._create_sequences(X_val, y_val, seq_len)
        
        # Build LSTM model
        model = keras.Sequential()
        
        for i, units in enumerate(LSTM_PARAMS['lstm_units']):
            return_sequences = i < len(LSTM_PARAMS['lstm_units']) - 1
            model.add(layers.LSTM(
                units,
                return_sequences=return_sequences,
                input_shape=(seq_len, X_train.shape[1]) if i == 0 else None
            ))
            model.add(layers.Dropout(LSTM_PARAMS['dropout']))
        
        model.add(layers.Dense(1, activation='sigmoid'))
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train
        history = model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=LSTM_PARAMS['epochs'],
            batch_size=LSTM_PARAMS['batch_size'],
            verbose=0
        )
        
        self.lstm_model = model
        
        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        
        logger.info(f"LSTM - Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}")
        
        return model
    
    def _create_sequences(self, X, y, seq_len):
        """Create sequences for LSTM."""
        X_seq, y_seq = [], []
        
        for i in range(len(X) - seq_len):
            X_seq.append(X[i:i+seq_len])
            y_seq.append(y[i+seq_len])
        
        return np.array(X_seq), np.array(y_seq)
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate both models on test set."""
        logger.info("Evaluating models on test set")
        
        results = {}
        
        # XGBoost evaluation
        if self.xgb_model:
            xgb_pred = self.xgb_model.predict(X_test)
            xgb_proba = self.xgb_model.predict_proba(X_test)[:, 1]
            
            results['xgboost'] = {
                'accuracy': (xgb_pred == y_test).mean(),
                'auc': roc_auc_score(y_test, xgb_proba),
                'predictions': xgb_pred,
                'probabilities': xgb_proba
            }
            
            logger.info(f"XGBoost Test - Accuracy: {results['xgboost']['accuracy']:.4f}, "
                       f"AUC: {results['xgboost']['auc']:.4f}")
        
        # LSTM evaluation
        if self.lstm_model:
            seq_len = LSTM_PARAMS['sequence_length']
            X_test_seq, y_test_seq = self._create_sequences(X_test, y_test, seq_len)
            
            lstm_proba = self.lstm_model.predict(X_test_seq).flatten()
            lstm_pred = (lstm_proba > 0.5).astype(int)
            
            results['lstm'] = {
                'accuracy': (lstm_pred == y_test_seq).mean(),
                'auc': roc_auc_score(y_test_seq, lstm_proba),
                'predictions': lstm_pred,
                'probabilities': lstm_proba
            }
            
            logger.info(f"LSTM Test - Accuracy: {results['lstm']['accuracy']:.4f}, "
                       f"AUC: {results['lstm']['auc']:.4f}")
        
        return results
    
    def plot_feature_importance(self, save_path: str):
        """Plot XGBoost feature importance."""
        if not self.xgb_model:
            return
        
        logger.info("Creating feature importance plot")
        
        importance = self.xgb_model.feature_importances_
        indices = np.argsort(importance)[::-1][:20]  # Top 20
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importance[indices])
        plt.yticks(range(len(indices)), [self.feature_columns[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Feature Importances (XGBoost)')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved feature importance plot to {save_path}")


def main():
    """Main function to train ML models."""
    logger.info("Starting ML model training")
    
    # Load data
    df = load_dataframe(REGIME_FILE, logger)
    
    # Initialize trainer
    trainer = MLTrainer()
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(df)
    
    # Train XGBoost
    trainer.train_xgboost(X_train, X_val, y_train, y_val)
    
    # Train LSTM
    trainer.train_lstm(X_train, X_val, y_train, y_val)
    
    # Evaluate models
    results = trainer.evaluate_models(X_test, y_test)
    
    # Save models
    save_model(trainer.xgb_model, XGBOOST_MODEL_FILE, logger)
    trainer.lstm_model.save(LSTM_MODEL_FILE)
    logger.info(f"Saved LSTM model to {LSTM_MODEL_FILE}")
    
    # Save scaler
    joblib.dump(trainer.scaler, SCALER_FILE)
    logger.info(f"Saved scaler to {SCALER_FILE}")
    
    # Plot feature importance
    trainer.plot_feature_importance(f'{RESULTS_DIR}/feature_importance.png')
    
    logger.info("ML model training completed successfully")


if __name__ == "__main__":
    main()
