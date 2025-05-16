import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from ..utils.config_loader import config
from ..utils.logger import get_logger

logger = get_logger(__name__)

class FollowerGrowthModel:
    """
    Model for predicting follower growth trends using time series analysis
    """
    
    def __init__(self):
        """
        Initialize the FollowerGrowthModel with configuration settings.
        """
        self.config = config.get_config("models").get("follower_growth", {})
        self.algorithm = self.config.get("algorithm", "lstm")
        self.params = self.config.get("params", {
            "units": 64,
            "dropout": 0.2,
            "recurrent_dropout": 0.2,
            "epochs": 50,
            "batch_size": 32
        })
        self.sequence_length = self.config.get("sequence_length", 30)
        self.feature_cols = self.config.get("features", [
            "post_frequency",
            "engagement_rate",
            "content_quality_score"
        ])
        
        # Initialize model and scalers
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.last_sequence = None
        self.last_date = None
    
    def train(self, df):
        """
        Train the follower growth model.
        
        Args:
            df (pandas.DataFrame): DataFrame with follower data
            
        Returns:
            dict: Training results
        """
        logger.info("Training follower growth model")
        
        # Check if required date column exists
        if "date" not in df.columns:
            logger.error("Date column not found in data")
            return {"success": False, "error": "Date column not found"}
        
        # Check if followers_count column exists
        if "followers_count" not in df.columns:
            logger.error("Followers count column not found in data")
            return {"success": False, "error": "Followers count column not found"}
        
        # Sort by date
        df = df.sort_values("date")
        
        # Save last date for forecasting
        self.last_date = df["date"].max()
        
        # Prepare features
        available_features = ["followers_count"]
        for col in self.feature_cols:
            if col in df.columns:
                available_features.append(col)
        
        logger.info(f"Using features: {available_features}")
        
        # Create sequences for time series prediction
        X, y = self._create_sequences(df, available_features)
        
        if len(X) == 0 or len(y) == 0:
            logger.error("Not enough data to create sequences")
            return {"success": False, "error": "Not enough data"}
        
        # Scale data
        X_scaled = self.scaler_X.fit_transform(X.reshape(X.shape[0], X.shape[1] * X.shape[2]))
        X_scaled = X_scaled.reshape(X.shape)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Save last sequence for forecasting
        self.last_sequence = X[-1:]
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
        
        # Build and train model
        if self.algorithm == "lstm":
            # Build LSTM model
            self.model = Sequential()
            self.model.add(LSTM(
                units=self.params.get("units", 64),
                dropout=self.params.get("dropout", 0.2),
                recurrent_dropout=self.params.get("recurrent_dropout", 0.2),
                return_sequences=True,
                input_shape=(X.shape[1], X.shape[2])
            ))
            self.model.add(LSTM(units=32))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(units=1))
            
            # Compile model
            self.model.compile(optimizer="adam", loss="mse")
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                epochs=self.params.get("epochs", 50),
                batch_size=self.params.get("batch_size", 32),
                validation_data=(X_test, y_test),
                verbose=0
            )
            
            # Evaluate model
            loss = history.history["loss"][-1]
            val_loss = history.history["val_loss"][-1]
            
            logger.info(f"Model trained with loss: {loss:.4f}, val_loss: {val_loss:.4f}")
            
            return {
                "success": True,
                "metrics": {
                    "loss": loss,
                    "val_loss": val_loss
                }
            }
        else:
            logger.error(f"Unsupported algorithm: {self.algorithm}")
            return {"success": False, "error": f"Unsupported algorithm: {self.algorithm}"}
    
    def _create_sequences(self, df, features):
        """
        Create sequences for time series prediction.
        
        Args:
            df (pandas.DataFrame): DataFrame with follower data
            features (list): List of feature columns
            
        Returns:
            tuple: X and y arrays for training
        """
        # Extract features
        data = df[features].values
        
        # Check if we have enough data
        if len(data) <= self.sequence_length:
            logger.warning(f"Not enough data for sequence length {self.sequence_length}")
            return np.array([]), np.array([])
        
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length, 0])  # followers_count is the target
        
        return np.array(X), np.array(y)
    
    def predict_growth(self, days=30):
        """
        Predict follower growth for the next several days.
        
        Args:
            days (int): Number of days to predict
            
        Returns:
            dict: Dictionary with predicted follower counts
        """
        if not self.model or self.last_sequence is None or self.last_date is None:
            logger.warning("Model not trained yet or missing data for prediction")
            return {}
        
        logger.info(f"Predicting follower growth for next {days} days")
        
        # Scale last sequence
        last_sequence_scaled = self.scaler_X.transform(
            self.last_sequence.reshape(self.last_sequence.shape[0], 
                                      self.last_sequence.shape[1] * self.last_sequence.shape[2])
        ).reshape(self.last_sequence.shape)
        
        # Make predictions
        predictions = []
        current_sequence = last_sequence_scaled.copy()
        
        for _ in range(days):
            # Predict next value
            next_val_scaled = self.model.predict(current_sequence, verbose=0)[0]
            
            # Inverse transform to get actual value
            next_val = self.scaler_y.inverse_transform([[next_val_scaled]])[0][0]
            predictions.append(next_val)
            
            # Update sequence for next prediction
            # Shift sequence and add new prediction
            new_row = current_sequence[0, 1:, :]
            new_prediction = np.zeros((1, 1, current_sequence.shape[2]))
            new_prediction[0, 0, 0] = next_val_scaled  # Set followers_count
            
            # Set other features to last values (simplified approach)
            for i in range(1, current_sequence.shape[2]):
                new_prediction[0, 0, i] = current_sequence[0, -1, i]
            
            current_sequence = np.concatenate([new_row, new_prediction], axis=1).reshape(1, self.sequence_length, -1)
        
        # Create result dictionary with dates
        result = {}
        current_date = pd.to_datetime(self.last_date)
        
        for i, pred in enumerate(predictions):
            current_date += timedelta(days=1)
            result[current_date.strftime("%Y-%m-%d")] = int(round(pred))
        
        return result 