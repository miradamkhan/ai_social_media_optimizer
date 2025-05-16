import os
import joblib
import numpy as np
from ..utils.config_loader import config
from ..utils.logger import get_logger
from .posting_time_model import PostingTimeModel
from .content_engagement_model import ContentEngagementModel
from .follower_growth_model import FollowerGrowthModel
from .retention_analysis_model import RetentionAnalysisModel

logger = get_logger(__name__)

class ModelManager:
    """
    Class for managing all machine learning models
    """
    
    def __init__(self, model_dir="models"):
        """
        Initialize the ModelManager.
        
        Args:
            model_dir (str): Directory to save/load models
        """
        self.model_dir = model_dir
        self.models_config = config.get_config("models")
        
        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize model instances
        self.posting_time_model = PostingTimeModel()
        self.content_engagement_model = ContentEngagementModel()
        self.follower_growth_model = FollowerGrowthModel()
        
        # Initialize retention analysis model if enabled
        retention_enabled = self.models_config.get("retention_analysis", {}).get("enabled", False)
        self.retention_analysis_model = RetentionAnalysisModel() if retention_enabled else None
        
        # Dictionary to track trained models
        self.trained_models = {
            "posting_time": False,
            "content_engagement": False,
            "follower_growth": False,
            "retention_analysis": False
        }
    
    def train_all_models(self, data_dict):
        """
        Train all models with the provided data.
        
        Args:
            data_dict (dict): Dictionary of DataFrames with engineered features
            
        Returns:
            dict: Dictionary of training results
        """
        logger.info("Training all models")
        
        results = {}
        
        # Train posting time model
        if "posting_time_optimization" in data_dict:
            logger.info("Training posting time model")
            posting_time_result = self.posting_time_model.train(data_dict["posting_time_optimization"])
            results["posting_time"] = posting_time_result
            self.trained_models["posting_time"] = True
        else:
            logger.warning("Posting time optimization data not available for training")
        
        # Train content engagement model
        if "posts" in data_dict:
            logger.info("Training content engagement model")
            content_engagement_result = self.content_engagement_model.train(data_dict["posts"])
            results["content_engagement"] = content_engagement_result
            self.trained_models["content_engagement"] = True
        else:
            logger.warning("Posts data not available for training content engagement model")
        
        # Train follower growth model
        if "followers" in data_dict:
            logger.info("Training follower growth model")
            follower_growth_result = self.follower_growth_model.train(data_dict["followers"])
            results["follower_growth"] = follower_growth_result
            self.trained_models["follower_growth"] = True
        else:
            logger.warning("Followers data not available for training follower growth model")
        
        # Train retention analysis model if enabled
        if self.retention_analysis_model and "retention_analysis" in data_dict:
            logger.info("Training retention analysis model")
            retention_result = self.retention_analysis_model.train(data_dict["retention_analysis"])
            results["retention_analysis"] = retention_result
            self.trained_models["retention_analysis"] = True
        elif self.retention_analysis_model:
            logger.warning("Retention analysis data not available for training")
        
        return results
    
    def save_models(self):
        """
        Save all trained models to disk.
        
        Returns:
            bool: True if successful
        """
        logger.info("Saving models to disk")
        
        try:
            # Save posting time model if trained
            if self.trained_models["posting_time"]:
                model_path = os.path.join(self.model_dir, "posting_time_model.pkl")
                joblib.dump(self.posting_time_model, model_path)
                logger.info(f"Saved posting time model to {model_path}")
            
            # Save content engagement model if trained
            if self.trained_models["content_engagement"]:
                model_path = os.path.join(self.model_dir, "content_engagement_model.pkl")
                joblib.dump(self.content_engagement_model, model_path)
                logger.info(f"Saved content engagement model to {model_path}")
            
            # Save follower growth model if trained
            if self.trained_models["follower_growth"]:
                model_path = os.path.join(self.model_dir, "follower_growth_model.pkl")
                joblib.dump(self.follower_growth_model, model_path)
                logger.info(f"Saved follower growth model to {model_path}")
            
            # Save retention analysis model if trained
            if self.trained_models["retention_analysis"] and self.retention_analysis_model:
                model_path = os.path.join(self.model_dir, "retention_analysis_model.pkl")
                joblib.dump(self.retention_analysis_model, model_path)
                logger.info(f"Saved retention analysis model to {model_path}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            return False
    
    def load_models(self):
        """
        Load all models from disk.
        
        Returns:
            bool: True if successful
        """
        logger.info("Loading models from disk")
        
        try:
            # Load posting time model if exists
            model_path = os.path.join(self.model_dir, "posting_time_model.pkl")
            if os.path.exists(model_path):
                self.posting_time_model = joblib.load(model_path)
                self.trained_models["posting_time"] = True
                logger.info(f"Loaded posting time model from {model_path}")
            
            # Load content engagement model if exists
            model_path = os.path.join(self.model_dir, "content_engagement_model.pkl")
            if os.path.exists(model_path):
                self.content_engagement_model = joblib.load(model_path)
                self.trained_models["content_engagement"] = True
                logger.info(f"Loaded content engagement model from {model_path}")
            
            # Load follower growth model if exists
            model_path = os.path.join(self.model_dir, "follower_growth_model.pkl")
            if os.path.exists(model_path):
                self.follower_growth_model = joblib.load(model_path)
                self.trained_models["follower_growth"] = True
                logger.info(f"Loaded follower growth model from {model_path}")
            
            # Load retention analysis model if exists
            model_path = os.path.join(self.model_dir, "retention_analysis_model.pkl")
            if os.path.exists(model_path):
                self.retention_analysis_model = joblib.load(model_path)
                self.trained_models["retention_analysis"] = True
                logger.info(f"Loaded retention analysis model from {model_path}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
    
    def get_optimal_posting_times(self, days=7):
        """
        Get optimal posting times for the next several days.
        
        Args:
            days (int): Number of days to predict
            
        Returns:
            dict: Dictionary of optimal posting times by day
        """
        if not self.trained_models["posting_time"]:
            logger.warning("Posting time model not trained")
            return {}
        
        return self.posting_time_model.predict_optimal_times(days)
    
    def get_content_recommendations(self):
        """
        Get content type recommendations.
        
        Returns:
            dict: Dictionary of content recommendations
        """
        if not self.trained_models["content_engagement"]:
            logger.warning("Content engagement model not trained")
            return {}
        
        return self.content_engagement_model.get_recommendations()
    
    def predict_follower_growth(self, days=30):
        """
        Predict follower growth for the next several days.
        
        Args:
            days (int): Number of days to predict
            
        Returns:
            dict: Dictionary with predicted follower counts
        """
        if not self.trained_models["follower_growth"]:
            logger.warning("Follower growth model not trained")
            return {}
        
        return self.follower_growth_model.predict_growth(days)
    
    def analyze_user_retention(self, data):
        """
        Analyze user retention and segment users.
        
        Args:
            data (pandas.DataFrame): User engagement data
            
        Returns:
            dict: Dictionary with retention analysis results
        """
        if not self.trained_models["retention_analysis"] or not self.retention_analysis_model:
            logger.warning("Retention analysis model not trained or not enabled")
            return {}
        
        return self.retention_analysis_model.analyze_retention(data) 