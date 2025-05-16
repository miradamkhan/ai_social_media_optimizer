import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from ..utils.config_loader import config
from ..utils.logger import get_logger

logger = get_logger(__name__)

class PostingTimeModel:
    """
    Model for predicting optimal posting times
    """
    
    def __init__(self):
        """
        Initialize the PostingTimeModel with configuration settings.
        """
        self.config = config.get_config("models").get("posting_time", {})
        self.algorithm = self.config.get("algorithm", "xgboost")
        self.params = self.config.get("params", {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1
        })
        self.feature_cols = self.config.get("features", [
            "hour_of_day",
            "day_of_week",
            "engagement_rate",
            "previous_post_performance"
        ])
        
        # Initialize model
        self.model = None
        self.feature_importances = {}
        self.best_times = {}
    
    def train(self, df):
        """
        Train the posting time model.
        
        Args:
            df (pandas.DataFrame): DataFrame with posting time features
            
        Returns:
            dict: Training results
        """
        logger.info("Training posting time model")
        
        # Check if required columns exist
        missing_cols = [col for col in self.feature_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            # Try to use available columns
            available_cols = [col for col in self.feature_cols if col in df.columns]
            if not available_cols:
                logger.error("No required columns available for training")
                return {"success": False, "error": "Missing required columns"}
            
            logger.info(f"Using available columns: {available_cols}")
            self.feature_cols = available_cols
        
        # Prepare data
        X = df[self.feature_cols]
        
        # Check for target column
        if "posting_time_score" in df.columns:
            y = df["posting_time_score"]
        elif "engagement_rate" in df.columns:
            y = df["engagement_rate"]
        else:
            # Create a default target based on hour_of_day if no target column exists
            logger.warning("No target column (posting_time_score or engagement_rate) found. Creating default target.")
            # Create a synthetic target based on hour of day - simple heuristic
            if "hour_of_day" in df.columns:
                # Assume business hours (9-17) are better for engagement
                business_hours = list(range(9, 18))
                df["synthetic_score"] = df["hour_of_day"].apply(
                    lambda h: 0.8 + (0.2 * (h in business_hours))
                )
                y = df["synthetic_score"]
            else:
                # If no hour_of_day column, use a constant target
                y = pd.Series([1.0] * len(df))
        
        # Handle missing values
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        if self.algorithm == "xgboost":
            self.model = xgb.XGBRegressor(
                n_estimators=self.params.get("n_estimators", 100),
                max_depth=self.params.get("max_depth", 5),
                learning_rate=self.params.get("learning_rate", 0.1),
                random_state=42
            )
            
            self.model.fit(X_train, y_train)
            
            # Get feature importances
            importances = self.model.feature_importances_
            self.feature_importances = {
                feature: importance
                for feature, importance in zip(self.feature_cols, importances)
            }
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"Model trained with MSE: {mse:.4f}, R²: {r2:.4f}")
            
            # Find best posting times
            self._find_best_posting_times(df)
            
            return {
                "success": True,
                "metrics": {
                    "mse": mse,
                    "r2": r2
                },
                "feature_importances": self.feature_importances
            }
        else:
            logger.error(f"Unsupported algorithm: {self.algorithm}")
            return {"success": False, "error": f"Unsupported algorithm: {self.algorithm}"}
    
    def _find_best_posting_times(self, df):
        """
        Find the best posting times based on the data.
        
        Args:
            df (pandas.DataFrame): DataFrame with posting time features
        """
        logger.info("Finding best posting times")
        
        # Group by day of week and hour of day
        if all(col in df.columns for col in ["day_of_week", "hour_of_day"]):
            # Determine which column to use for scoring
            if "posting_time_score" in df.columns:
                score_col = "posting_time_score"
            elif "engagement_rate" in df.columns:
                score_col = "engagement_rate"
            elif "synthetic_score" in df.columns:
                score_col = "synthetic_score"
            else:
                # Create a default score based on hour of day
                business_hours = list(range(9, 18))
                df["default_score"] = df["hour_of_day"].apply(
                    lambda h: 0.8 + (0.2 * (h in business_hours))
                )
                score_col = "default_score"
            
            time_performance = df.groupby(["day_of_week", "hour_of_day"])[score_col].mean().reset_index()
            
            # Find best times for each day
            for day in range(7):
                day_data = time_performance[time_performance["day_of_week"] == day]
                if not day_data.empty:
                    # Get top 3 hours
                    top_hours = day_data.sort_values(score_col, ascending=False).head(3)
                    self.best_times[day] = top_hours["hour_of_day"].tolist()
                else:
                    self.best_times[day] = []
            
            logger.info(f"Best posting times found: {self.best_times}")
        else:
            logger.warning("Cannot find best posting times: missing day_of_week or hour_of_day columns")
            # Default to business hours
            for day in range(7):
                self.best_times[day] = [9, 12, 17]  # Default to morning, noon, and afternoon
    
    def predict_optimal_times(self, days=7):
        """
        Predict optimal posting times for the next several days.
        
        Args:
            days (int): Number of days to predict
            
        Returns:
            dict: Dictionary of optimal posting times by day
        """
        if not self.model:
            logger.warning("Model not trained yet")
            return {}
        
        logger.info(f"Predicting optimal posting times for next {days} days")
        
        result = {}
        today = datetime.now()
        
        for i in range(days):
            target_date = today + timedelta(days=i)
            day_of_week = target_date.weekday()
            
            # Get best hours for this day of week
            if day_of_week in self.best_times and self.best_times[day_of_week]:
                best_hours = self.best_times[day_of_week]
                
                # Convert hours to datetime
                optimal_times = [
                    target_date.replace(hour=hour, minute=0, second=0, microsecond=0).strftime("%Y-%m-%d %H:%M:%S")
                    for hour in best_hours
                ]
                
                result[target_date.strftime("%Y-%m-%d")] = optimal_times
            else:
                # Default to business hours if no data
                default_hours = [9, 12, 17]
                optimal_times = [
                    target_date.replace(hour=hour, minute=0, second=0, microsecond=0).strftime("%Y-%m-%d %H:%M:%S")
                    for hour in default_hours
                ]
                
                result[target_date.strftime("%Y-%m-%d")] = optimal_times
        
        return result
    
    def predict_engagement(self, features_df):
        """
        Predict engagement for given features.
        
        Args:
            features_df (pandas.DataFrame): DataFrame with features
            
        Returns:
            numpy.ndarray: Predicted engagement scores
        """
        if not self.model:
            logger.warning("Model not trained yet")
            return np.array([])
        
        # Check if required columns exist
        missing_cols = [col for col in self.feature_cols if col not in features_df.columns]
        if missing_cols:
            logger.warning(f"Missing required columns for prediction: {missing_cols}")
            return np.array([])
        
        # Prepare features
        X = features_df[self.feature_cols]
        X = X.fillna(X.mean())
        
        # Make prediction
        return self.model.predict(X) 