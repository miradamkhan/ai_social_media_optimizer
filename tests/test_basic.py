import os
import sys
import unittest
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_ingestion import DataLoader
from src.preprocessing import DataPreprocessor, FeatureEngineer
from src.models import ModelManager

class TestBasic(unittest.TestCase):
    """Basic tests for the Social Media Analytics Tool."""
    
    def setUp(self):
        """Set up test environment."""
        # Create sample data
        self.sample_posts = pd.DataFrame({
            'post_id': ['post_1', 'post_2', 'post_3'],
            'timestamp': pd.to_datetime(['2023-06-01 09:00:00', '2023-06-02 12:00:00', '2023-06-03 15:00:00']),
            'content_type': ['image', 'video', 'carousel'],
            'caption_length': [100, 150, 200],
            'hashtags': [5, 7, 4],
            'mentions': [2, 1, 3],
            'likes': [500, 800, 600],
            'comments': [50, 70, 60],
            'shares': [25, 40, 30],
            'saves': [15, 25, 20],
            'reach': [2500, 4000, 3000],
            'impressions': [3000, 5000, 3500]
        })
        
        self.sample_followers = pd.DataFrame({
            'date': pd.to_datetime(['2023-06-01', '2023-06-02', '2023-06-03']),
            'followers_count': [10000, 10050, 10100],
            'followers_gained': [60, 70, 80],
            'followers_lost': [10, 20, 30],
            'profile_views': [500, 550, 600],
            'reach': [3000, 3200, 3400]
        })
        
        self.sample_engagement = pd.DataFrame({
            'timestamp': pd.to_datetime(['2023-06-01 10:00:00', '2023-06-01 14:00:00', '2023-06-02 10:00:00']),
            'active_users': [300, 400, 350],
            'likes_received': [80, 100, 90],
            'comments_received': [15, 20, 18],
            'shares_received': [8, 10, 9],
            'story_views': [450, 550, 500],
            'profile_clicks': [40, 50, 45]
        })
        
        # Create sample data dictionary
        self.sample_data = {
            'posts': self.sample_posts,
            'followers': self.sample_followers,
            'engagement': self.sample_engagement
        }
    
    def test_preprocessing(self):
        """Test data preprocessing."""
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.preprocess(self.sample_data)
        
        # Check if processed data has the same keys
        self.assertEqual(set(processed_data.keys()), set(self.sample_data.keys()))
        
        # Check if processed data has the same number of rows
        for key in self.sample_data:
            self.assertEqual(len(processed_data[key]), len(self.sample_data[key]))
    
    def test_feature_engineering(self):
        """Test feature engineering."""
        feature_engineer = FeatureEngineer()
        engineered_data = feature_engineer.engineer_features(self.sample_data)
        
        # Check if engineered data has at least the same keys
        self.assertTrue(set(self.sample_data.keys()).issubset(set(engineered_data.keys())))
        
        # Check if posts data has new features
        if 'posts' in engineered_data:
            self.assertTrue('hour_of_day' in engineered_data['posts'].columns)
            self.assertTrue('day_of_week' in engineered_data['posts'].columns)
    
    def test_model_initialization(self):
        """Test model initialization."""
        model_manager = ModelManager()
        
        # Check if models are initialized
        self.assertIsNotNone(model_manager.posting_time_model)
        self.assertIsNotNone(model_manager.content_engagement_model)
        self.assertIsNotNone(model_manager.follower_growth_model)


if __name__ == '__main__':
    unittest.main() 