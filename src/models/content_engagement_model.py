import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from ..utils.config_loader import config
from ..utils.logger import get_logger

logger = get_logger(__name__)

class ContentEngagementModel:
    """
    Model for predicting content engagement and recommending content types
    """
    
    def __init__(self):
        """
        Initialize the ContentEngagementModel with configuration settings.
        """
        self.config = config.get_config("models").get("content_engagement", {})
        self.algorithm = self.config.get("algorithm", "random_forest")
        self.params = self.config.get("params", {
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 5
        })
        self.feature_cols = self.config.get("features", [
            "content_type",
            "content_length",
            "hashtags",
            "mentions",
            "media_count"
        ])
        
        # Initialize model
        self.model = None
        self.feature_importances = {}
        self.content_performance = {}
    
    def train(self, df):
        """
        Train the content engagement model.
        
        Args:
            df (pandas.DataFrame): DataFrame with content features
            
        Returns:
            dict: Training results
        """
        logger.info("Training content engagement model")
        
        # Make a copy of the data to avoid modifying the original
        df = df.copy()
        
        # Check for duplicate columns and drop them
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Check if required columns exist
        available_cols = [col for col in self.feature_cols if col in df.columns]
        if not available_cols:
            logger.error("No required columns available for training")
            return {"success": False, "error": "Missing required columns"}
        
        # Handle content_type column
        if "content_type" in available_cols:
            # One-hot encode content type
            if "content_type" in df.columns:
                content_dummies = pd.get_dummies(df["content_type"], prefix="content")
                df = pd.concat([df, content_dummies], axis=1)
            
            # Remove content_type from available columns
            available_cols.remove("content_type")
            
            # Add the one-hot encoded columns to available_cols
            content_type_cols = [col for col in df.columns if col.startswith("content_") and col != "content_type"]
            available_cols.extend(content_type_cols)
        else:
            # Check if one-hot encoded content columns already exist
            content_type_cols = [col for col in df.columns if col.startswith("content_") and col != "content_type"]
            if content_type_cols:
                available_cols.extend([col for col in content_type_cols if col not in available_cols])
        
        # If no features, return error
        if not available_cols:
            logger.error("No features available for training")
            return {"success": False, "error": "No features available"}
        
        # Remove duplicates from available_cols
        available_cols = list(dict.fromkeys(available_cols))
        
        logger.info(f"Using features: {available_cols}")
        
        # Check for duplicate columns again after potential concat operation
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Prepare data
        X = df[available_cols].copy()
        y = df["engagement_rate"].copy() if "engagement_rate" in df.columns else df["likes"].copy()
        
        # Handle missing values - separately for numeric and non-numeric columns
        for col in X.columns:
            if pd.api.types.is_object_dtype(X[col]):  # For string/categorical columns
                X.loc[:, col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "unknown")
            else:  # For numeric columns
                X.loc[:, col] = X[col].fillna(X[col].mean())
                
        y = y.fillna(y.mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        if self.algorithm == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=self.params.get("n_estimators", 200),
                max_depth=self.params.get("max_depth", 10),
                min_samples_split=self.params.get("min_samples_split", 5),
                random_state=42
            )
            
            self.model.fit(X_train, y_train)
            
            # Get feature importances
            importances = self.model.feature_importances_
            self.feature_importances = {
                feature: importance
                for feature, importance in zip(available_cols, importances)
            }
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"Model trained with MSE: {mse:.4f}, R²: {r2:.4f}")
            
            # Analyze content type performance
            content_type_cols = [col for col in df.columns if col.startswith("content_") and col != "content_type"]
            self._analyze_content_performance(df, content_type_cols)
            
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
    
    def _analyze_content_performance(self, df, content_type_cols):
        """
        Analyze content type performance.
        
        Args:
            df (pandas.DataFrame): DataFrame with content features
            content_type_cols (list): List of content type columns
        """
        logger.info("Analyzing content type performance")
        
        if not content_type_cols:
            logger.warning("No content type columns available for analysis")
            return
        
        # Get engagement metric
        engagement_col = "engagement_rate" if "engagement_rate" in df.columns else "likes"
        
        # Calculate average engagement by content type
        for col in content_type_cols:
            content_type = col.replace("content_", "")
            content_data = df[df[col] == 1]
            
            if not content_data.empty:
                avg_engagement = content_data[engagement_col].mean()
                self.content_performance[content_type] = avg_engagement
        
        # Sort by performance
        self.content_performance = {
            k: v for k, v in sorted(
                self.content_performance.items(),
                key=lambda item: item[1],
                reverse=True
            )
        }
        
        logger.info(f"Content performance analysis: {self.content_performance}")
    
    def get_recommendations(self):
        """
        Get content type recommendations.
        
        Returns:
            dict: Dictionary of content recommendations
        """
        if not self.model or not self.content_performance:
            logger.warning("Model not trained yet or no content performance data")
            return {}
        
        logger.info("Generating content recommendations")
        
        # Create recommendations
        recommendations = {
            "top_content_types": list(self.content_performance.keys())[:3],
            "content_performance": self.content_performance,
            "feature_importance": self.feature_importances
        }
        
        return recommendations
    
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
            
        # Make a copy to avoid modifying the original
        features_df = features_df.copy()
        
        # Check for duplicate columns and drop them
        features_df = features_df.loc[:, ~features_df.columns.duplicated()]
        
        # Handle content_type column
        content_type_cols = []
        if "content_type" in features_df.columns:
            # One-hot encode content type
            content_dummies = pd.get_dummies(features_df["content_type"], prefix="content")
            features_df = pd.concat([features_df, content_dummies], axis=1)
            # Check for duplicate columns again
            features_df = features_df.loc[:, ~features_df.columns.duplicated()]
            content_type_cols = [col for col in features_df.columns if col.startswith("content_") and col != "content_type"]
        else:
            # Check if one-hot encoded content columns already exist
            content_type_cols = [col for col in features_df.columns if col.startswith("content_") and col != "content_type"]
        
        # Get feature columns, excluding content_type in favor of one-hot encoded columns
        feature_cols = []
        for col in self.feature_cols:
            if col == "content_type" and content_type_cols:
                # Add one-hot encoded columns instead of original
                feature_cols.extend(content_type_cols)
            elif col != "content_type" and col in features_df.columns:
                feature_cols.append(col)
        
        # Remove duplicates
        feature_cols = list(dict.fromkeys(feature_cols))
        
        # Check if we have enough features
        if not feature_cols:
            logger.warning("No features available for prediction")
            return np.array([])
        
        # Prepare features
        X = features_df[feature_cols].copy()
        
        # Handle missing values - separately for numeric and non-numeric columns
        for col in X.columns:
            if pd.api.types.is_object_dtype(X[col]):  # For string/categorical columns
                X.loc[:, col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "unknown")
            else:  # For numeric columns
                X.loc[:, col] = X[col].fillna(X[col].mean())
        
        # Make prediction
        return self.model.predict(X) 