import pandas as pd
import numpy as np
from datetime import datetime
from ..utils.config_loader import config
from ..utils.logger import get_logger

logger = get_logger(__name__)

class FeatureEngineer:
    """
    Class for engineering features from social media data
    """
    
    def __init__(self):
        """
        Initialize the FeatureEngineer with configuration settings.
        """
        self.models_config = config.get_config("models")
    
    def engineer_features(self, data_dict):
        """
        Engineer features from the data dictionary.
        
        Args:
            data_dict (dict): Dictionary of DataFrames with keys for each data type
            
        Returns:
            dict: Dictionary of DataFrames with engineered features
        """
        logger.info("Starting feature engineering")
        
        result = {}
        
        # Engineer features for each data type
        if "posts" in data_dict:
            result["posts"] = self._engineer_post_features(data_dict["posts"])
        
        if "followers" in data_dict:
            result["followers"] = self._engineer_follower_features(data_dict["followers"])
        
        if "engagement" in data_dict:
            result["engagement"] = self._engineer_engagement_features(data_dict["engagement"])
        
        # Create combined features if multiple data types are available
        if len(data_dict) > 1:
            result.update(self._engineer_combined_features(data_dict))
        
        return result
    
    def _engineer_post_features(self, posts_df):
        """
        Engineer features from posts data.
        
        Args:
            posts_df (pandas.DataFrame): Posts DataFrame
            
        Returns:
            pandas.DataFrame: DataFrame with engineered features
        """
        logger.info("Engineering post features")
        
        # Make a copy to avoid modifying the original
        df = posts_df.copy()
        
        # Ensure timestamp is datetime
        if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Extract time-based features
        if "timestamp" in df.columns:
            df["hour_of_day"] = df["timestamp"].dt.hour
            df["day_of_week"] = df["timestamp"].dt.dayofweek
            df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
            df["month"] = df["timestamp"].dt.month
            df["quarter"] = df["timestamp"].dt.quarter
            
            # Time of day category
            df["time_of_day"] = pd.cut(
                df["hour_of_day"],
                bins=[0, 6, 12, 18, 24],
                labels=["night", "morning", "afternoon", "evening"],
                include_lowest=True
            )
        
        # Calculate engagement metrics
        if all(col in df.columns for col in ["likes", "comments"]):
            df["engagement_count"] = df["likes"] + df["comments"]
            
            if "shares" in df.columns:
                df["engagement_count"] += df["shares"]
                
            if "saves" in df.columns:
                df["engagement_count"] += df["saves"]
        
        # Calculate engagement rate if reach or impressions available
        if "engagement_count" in df.columns:
            if "reach" in df.columns and df["reach"].sum() > 0:
                df["engagement_rate"] = df["engagement_count"] / df["reach"]
            elif "impressions" in df.columns and df["impressions"].sum() > 0:
                df["engagement_rate"] = df["engagement_count"] / df["impressions"]
        
        # One-hot encode content type
        if "content_type" in df.columns:
            content_type_dummies = pd.get_dummies(df["content_type"], prefix="content")
            df = pd.concat([df, content_type_dummies], axis=1)
        
        # Calculate post frequency features
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp")
            df["time_since_last_post"] = df["timestamp"].diff().dt.total_seconds() / 3600  # hours
            
            # Rolling average of engagement
            if "engagement_rate" in df.columns:
                df["engagement_rate_rolling_avg"] = df["engagement_rate"].rolling(window=3, min_periods=1).mean()
                df["engagement_rate_rolling_std"] = df["engagement_rate"].rolling(window=3, min_periods=1).std()
        
        # Calculate content quality score (example metric)
        if all(col in df.columns for col in ["likes", "comments", "shares"]):
            df["content_quality_score"] = (
                df["likes"] * 1 +
                df["comments"] * 2 +
                df["shares"] * 3
            ) / (df["likes"] + df["comments"] + df["shares"]).clip(lower=1)
        
        logger.info(f"Engineered post features: {df.shape}")
        return df
    
    def _engineer_follower_features(self, followers_df):
        """
        Engineer features from followers data.
        
        Args:
            followers_df (pandas.DataFrame): Followers DataFrame
            
        Returns:
            pandas.DataFrame: DataFrame with engineered features
        """
        logger.info("Engineering follower features")
        
        # Make a copy to avoid modifying the original
        df = followers_df.copy()
        
        # Ensure date is datetime
        if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"])
        
        # Sort by date
        if "date" in df.columns:
            df = df.sort_values("date")
        
        # Calculate growth rate
        if "followers_count" in df.columns:
            df["followers_growth_rate"] = df["followers_count"].pct_change().fillna(0)
            
            # Rolling metrics
            df["followers_growth_rate_rolling_avg"] = df["followers_growth_rate"].rolling(window=7, min_periods=1).mean()
            df["followers_growth_rate_rolling_std"] = df["followers_growth_rate"].rolling(window=7, min_periods=1).std()
            
            # Cumulative growth
            first_value = df["followers_count"].iloc[0] if not df.empty else 0
            if first_value > 0:
                df["cumulative_growth_pct"] = (df["followers_count"] - first_value) / first_value
        
        # Calculate net growth
        if all(col in df.columns for col in ["followers_gained", "followers_lost"]):
            df["net_follower_change"] = df["followers_gained"] - df["followers_lost"]
            df["follower_retention_rate"] = 1 - (df["followers_lost"] / df["followers_count"].shift(1)).fillna(0)
        
        # Extract time-based features
        if "date" in df.columns:
            df["day_of_week"] = df["date"].dt.dayofweek
            df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
            df["month"] = df["date"].dt.month
            df["quarter"] = df["date"].dt.quarter
        
        # Engagement efficiency
        if all(col in df.columns for col in ["profile_views", "followers_gained"]):
            df["conversion_rate"] = df["followers_gained"] / df["profile_views"].clip(lower=1)
        
        logger.info(f"Engineered follower features: {df.shape}")
        return df
    
    def _engineer_engagement_features(self, engagement_df):
        """
        Engineer features from engagement data.
        
        Args:
            engagement_df (pandas.DataFrame): Engagement DataFrame
            
        Returns:
            pandas.DataFrame: DataFrame with engineered features
        """
        logger.info("Engineering engagement features")
        
        # Make a copy to avoid modifying the original
        df = engagement_df.copy()
        
        # Ensure timestamp is datetime
        if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Extract time-based features
        if "timestamp" in df.columns:
            df["hour_of_day"] = df["timestamp"].dt.hour
            df["day_of_week"] = df["timestamp"].dt.dayofweek
            df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
            df["date"] = df["timestamp"].dt.date
            
            # Time of day category
            df["time_of_day"] = pd.cut(
                df["hour_of_day"],
                bins=[0, 6, 12, 18, 24],
                labels=["night", "morning", "afternoon", "evening"],
                include_lowest=True
            )
        
        # Calculate total engagement
        engagement_cols = [col for col in df.columns if any(term in col for term in ["likes", "comments", "shares", "views"])]
        if engagement_cols:
            df["total_engagement"] = df[engagement_cols].sum(axis=1)
        
        # Calculate engagement rate
        if "total_engagement" in df.columns and "active_users" in df.columns:
            df["engagement_rate"] = df["total_engagement"] / df["active_users"].clip(lower=1)
        
        # Aggregate by date and hour for time patterns
        if "timestamp" in df.columns:
            # Group by hour to find peak engagement times
            hourly_engagement = df.groupby("hour_of_day")["total_engagement"].mean().reset_index()
            peak_hours = hourly_engagement.sort_values("total_engagement", ascending=False)["hour_of_day"].tolist()
            
            if peak_hours:
                df["is_peak_hour"] = df["hour_of_day"].isin(peak_hours[:3]).astype(int)
        
        # Calculate interaction depth
        if "likes_received" in df.columns and "comments_received" in df.columns:
            df["interaction_depth"] = df["comments_received"] / (df["likes_received"] + df["comments_received"]).clip(lower=1)
        
        logger.info(f"Engineered engagement features: {df.shape}")
        return df
    
    def _engineer_combined_features(self, data_dict):
        """
        Engineer features that combine multiple data types.
        
        Args:
            data_dict (dict): Dictionary of DataFrames with keys for each data type
            
        Returns:
            dict: Dictionary of DataFrames with combined features
        """
        logger.info("Engineering combined features")
        
        result = {}
        
        # Create posting time optimization features
        if "posts" in data_dict and "engagement" in data_dict:
            posts_df = data_dict["posts"]
            engagement_df = data_dict["engagement"]
            
            # Ensure timestamp columns are datetime
            if "timestamp" in posts_df.columns and not pd.api.types.is_datetime64_any_dtype(posts_df["timestamp"]):
                posts_df["timestamp"] = pd.to_datetime(posts_df["timestamp"])
                
            if "timestamp" in engagement_df.columns and not pd.api.types.is_datetime64_any_dtype(engagement_df["timestamp"]):
                engagement_df["timestamp"] = pd.to_datetime(engagement_df["timestamp"])
            
            # Extract hour and day of week
            if "timestamp" in posts_df.columns:
                posts_df["hour_of_day"] = posts_df["timestamp"].dt.hour
                posts_df["day_of_week"] = posts_df["timestamp"].dt.dayofweek
                
            if "timestamp" in engagement_df.columns:
                engagement_df["hour_of_day"] = engagement_df["timestamp"].dt.hour
                engagement_df["day_of_week"] = engagement_df["timestamp"].dt.dayofweek
            
            # Ensure total_engagement column exists
            if "total_engagement" not in engagement_df.columns:
                # Calculate total engagement
                engagement_cols = [col for col in engagement_df.columns if any(term in col for term in ["likes", "comments", "shares", "views"])]
                if engagement_cols:
                    engagement_df["total_engagement"] = engagement_df[engagement_cols].sum(axis=1)
                else:
                    # Create a default total_engagement column if no engagement columns exist
                    engagement_df["total_engagement"] = 1
            
            # Aggregate engagement by hour and day of week
            if all(col in engagement_df.columns for col in ["hour_of_day", "day_of_week"]):
                hourly_engagement = engagement_df.groupby(["hour_of_day", "day_of_week"])["total_engagement"].mean().reset_index()
                hourly_engagement.rename(columns={"total_engagement": "avg_engagement_by_time"}, inplace=True)
                
                # Merge with posts data
                if all(col in posts_df.columns for col in ["hour_of_day", "day_of_week"]):
                    posting_time_df = posts_df.merge(
                        hourly_engagement,
                        on=["hour_of_day", "day_of_week"],
                        how="left"
                    )
                    
                    # Calculate optimal posting time score
                    if "engagement_rate" in posting_time_df.columns and "avg_engagement_by_time" in posting_time_df.columns:
                        posting_time_df["posting_time_score"] = (
                            posting_time_df["engagement_rate"] * 
                            posting_time_df["avg_engagement_by_time"].fillna(posting_time_df["avg_engagement_by_time"].mean())
                        )
                    
                    result["posting_time_optimization"] = posting_time_df
        
        # Create content performance features
        if "posts" in data_dict and "followers" in data_dict:
            posts_df = data_dict["posts"]
            followers_df = data_dict["followers"]
            
            # Ensure date columns are datetime
            if "timestamp" in posts_df.columns and not pd.api.types.is_datetime64_any_dtype(posts_df["timestamp"]):
                posts_df["timestamp"] = pd.to_datetime(posts_df["timestamp"])
                posts_df["date"] = posts_df["timestamp"].dt.date
                
            if "date" in followers_df.columns and not pd.api.types.is_datetime64_any_dtype(followers_df["date"]):
                followers_df["date"] = pd.to_datetime(followers_df["date"]).dt.date
            
            # Aggregate posts by date
            if "date" in posts_df.columns and "content_type" in posts_df.columns:
                daily_content = posts_df.groupby(["date", "content_type"]).agg({
                    "engagement_rate": "mean",
                    "post_id": "count"
                }).reset_index()
                daily_content.rename(columns={"post_id": "post_count"}, inplace=True)
                
                # Merge with followers data
                if "date" in followers_df.columns:
                    content_performance_df = daily_content.merge(
                        followers_df[["date", "followers_gained"]],
                        on="date",
                        how="left"
                    )
                    
                    # Calculate content impact score
                    content_performance_df["content_impact_score"] = (
                        content_performance_df["engagement_rate"] * 
                        content_performance_df["followers_gained"].fillna(0)
                    )
                    
                    result["content_performance"] = content_performance_df
        
        # Create retention analysis features
        if "engagement" in data_dict and "followers" in data_dict:
            engagement_df = data_dict["engagement"]
            followers_df = data_dict["followers"]
            
            # Ensure date columns are datetime
            if "timestamp" in engagement_df.columns and not pd.api.types.is_datetime64_any_dtype(engagement_df["timestamp"]):
                engagement_df["timestamp"] = pd.to_datetime(engagement_df["timestamp"])
                engagement_df["date"] = engagement_df["timestamp"].dt.date
                
            if "date" in followers_df.columns and not pd.api.types.is_datetime64_any_dtype(followers_df["date"]):
                followers_df["date"] = pd.to_datetime(followers_df["date"]).dt.date
            
            # Ensure total_engagement column exists in engagement_df
            if "total_engagement" not in engagement_df.columns:
                engagement_cols = [col for col in engagement_df.columns if any(term in col for term in ["likes", "comments", "shares", "views"])]
                if engagement_cols:
                    engagement_df["total_engagement"] = engagement_df[engagement_cols].sum(axis=1)
                else:
                    # Create a default total_engagement column if no engagement columns exist
                    engagement_df["total_engagement"] = 1
            
            # Aggregate engagement by date
            if "date" in engagement_df.columns:
                daily_engagement = engagement_df.groupby("date").agg({
                    "active_users": "mean",
                    "total_engagement": "sum"
                }).reset_index()
                
                # Merge with followers data
                if "date" in followers_df.columns:
                    retention_df = daily_engagement.merge(
                        followers_df[["date", "followers_count", "followers_lost"]],
                        on="date",
                        how="left"
                    )
                    
                    # Calculate retention metrics
                    retention_df["engagement_per_follower"] = (
                        retention_df["total_engagement"] / 
                        retention_df["followers_count"].fillna(1)
                    )
                    
                    retention_df["retention_score"] = 1 - (
                        retention_df["followers_lost"].fillna(0) / 
                        retention_df["followers_count"].shift(1).fillna(retention_df["followers_count"].iloc[0] if not retention_df.empty else 1)
                    )
                    
                    result["retention_analysis"] = retention_df
        
        logger.info(f"Engineered combined features: {len(result)} datasets")
        return result 