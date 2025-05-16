import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from ..utils.config_loader import config
from ..utils.logger import get_logger

logger = get_logger(__name__)

class APIClient:
    """
    Client for fetching data from social media platform APIs
    """
    
    def __init__(self):
        """
        Initialize the API client with API keys from environment variables.
        """
        # Load API keys from environment variables
        self.api_keys = {
            "instagram": {
                "api_key": config.get_env_var("INSTAGRAM_API_KEY"),
                "api_secret": config.get_env_var("INSTAGRAM_API_SECRET")
            },
            "tiktok": {
                "api_key": config.get_env_var("TIKTOK_API_KEY"),
                "api_secret": config.get_env_var("TIKTOK_API_SECRET")
            },
            "twitter": {
                "api_key": config.get_env_var("TWITTER_API_KEY"),
                "api_secret": config.get_env_var("TWITTER_API_SECRET"),
                "bearer_token": config.get_env_var("TWITTER_BEARER_TOKEN")
            }
        }
        
        # Load configuration
        self.config = config.get_config("data_ingestion")
        self.batch_size = self.config.get("batch_size", 1000)
        self.max_posts = self.config.get("max_posts_per_fetch", 5000)
    
    def fetch_data(self, platform, data_type=None, start_date=None, end_date=None):
        """
        Fetch data from the specified platform's API.
        
        Args:
            platform (str): Social media platform to fetch data from
            data_type (str, optional): Type of data to fetch (posts, followers, engagement)
            start_date (str, optional): Start date for data filtering (YYYY-MM-DD)
            end_date (str, optional): End date for data filtering (YYYY-MM-DD)
            
        Returns:
            dict: Dictionary of DataFrames with keys for each data type
        """
        if platform not in self.api_keys:
            logger.error(f"Unsupported platform: {platform}")
            return {}
            
        # Check if API keys are available
        if not all(self.api_keys[platform].values()):
            logger.error(f"Missing API keys for platform: {platform}")
            return {}
            
        # Parse dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else datetime.now() - timedelta(days=30)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()
        
        # Select the appropriate fetch method based on platform
        if platform == "instagram":
            return self._fetch_instagram_data(data_type, start_dt, end_dt)
        elif platform == "tiktok":
            return self._fetch_tiktok_data(data_type, start_dt, end_dt)
        elif platform == "twitter":
            return self._fetch_twitter_data(data_type, start_dt, end_dt)
        else:
            logger.error(f"No fetch method implemented for platform: {platform}")
            return {}
    
    def _fetch_instagram_data(self, data_type, start_date, end_date):
        """
        Fetch data from Instagram API.
        
        Args:
            data_type (str): Type of data to fetch
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            dict: Dictionary of DataFrames
        """
        logger.info(f"Fetching Instagram data: {data_type} from {start_date} to {end_date}")
        
        # In a real implementation, this would use the Instagram Graph API
        # For now, we'll simulate the API response
        
        result = {}
        
        # Fetch posts data if requested or if no specific type
        if data_type in [None, "posts"]:
            posts_data = self._simulate_instagram_posts(start_date, end_date)
            result["posts"] = pd.DataFrame(posts_data)
        
        # Fetch followers data if requested or if no specific type
        if data_type in [None, "followers"]:
            followers_data = self._simulate_instagram_followers(start_date, end_date)
            result["followers"] = pd.DataFrame(followers_data)
        
        # Fetch engagement data if requested or if no specific type
        if data_type in [None, "engagement"]:
            engagement_data = self._simulate_instagram_engagement(start_date, end_date)
            result["engagement"] = pd.DataFrame(engagement_data)
        
        return result
    
    def _fetch_tiktok_data(self, data_type, start_date, end_date):
        """
        Fetch data from TikTok API.
        
        Args:
            data_type (str): Type of data to fetch
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            dict: Dictionary of DataFrames
        """
        logger.info(f"Fetching TikTok data: {data_type} from {start_date} to {end_date}")
        
        # Similar to Instagram, this would use the TikTok API
        # For now, we'll simulate the API response
        
        # Implementation would be similar to Instagram but with TikTok-specific fields
        # Placeholder for now
        return {}
    
    def _fetch_twitter_data(self, data_type, start_date, end_date):
        """
        Fetch data from Twitter API.
        
        Args:
            data_type (str): Type of data to fetch
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            dict: Dictionary of DataFrames
        """
        logger.info(f"Fetching Twitter data: {data_type} from {start_date} to {end_date}")
        
        # Similar to Instagram, this would use the Twitter API
        # For now, we'll simulate the API response
        
        # Implementation would be similar to Instagram but with Twitter-specific fields
        # Placeholder for now
        return {}
    
    def _simulate_instagram_posts(self, start_date, end_date):
        """
        Simulate Instagram posts data for development purposes.
        
        Args:
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            list: List of post dictionaries
        """
        # Simulate API rate limiting
        time.sleep(0.1)
        
        # Generate sample data
        posts = []
        current_date = start_date
        
        while current_date <= end_date and len(posts) < self.max_posts:
            # Generate 1-3 posts per day
            for _ in range(1, 4):
                if len(posts) >= self.max_posts:
                    break
                    
                # Random hour of the day
                hour = 8 + ((_ * 5) % 12)  # Spread posts throughout the day
                post_time = current_date.replace(hour=hour)
                
                # Alternate between different content types
                content_type = ["image", "video", "carousel"][_ % 3]
                
                posts.append({
                    "post_id": f"post_{len(posts) + 1}",
                    "timestamp": post_time.isoformat(),
                    "content_type": content_type,
                    "caption_length": 50 + (len(posts) % 200),
                    "hashtags": 3 + (len(posts) % 7),
                    "mentions": 1 + (len(posts) % 3),
                    "likes": 100 + (len(posts) * 10) % 1000,
                    "comments": 10 + (len(posts) * 5) % 100,
                    "shares": 5 + (len(posts) * 2) % 50,
                    "saves": 2 + (len(posts) * 3) % 30,
                    "reach": 500 + (len(posts) * 50) % 5000,
                    "impressions": 700 + (len(posts) * 70) % 7000,
                })
            
            # Move to next day
            current_date += timedelta(days=1)
        
        return posts
    
    def _simulate_instagram_followers(self, start_date, end_date):
        """
        Simulate Instagram followers data for development purposes.
        
        Args:
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            list: List of follower dictionaries
        """
        # Simulate API rate limiting
        time.sleep(0.1)
        
        # Generate sample data
        followers_data = []
        current_date = start_date
        base_followers = 10000
        
        while current_date <= end_date:
            # Add some randomness to follower growth
            daily_growth = 50 + (current_date.day * 2) % 100
            base_followers += daily_growth
            
            followers_data.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "followers_count": base_followers,
                "followers_gained": daily_growth,
                "followers_lost": 10 + (current_date.day % 20),
                "profile_views": 500 + (current_date.day * 30) % 1000,
                "reach": 2000 + (current_date.day * 100) % 5000
            })
            
            # Move to next day
            current_date += timedelta(days=1)
        
        return followers_data
    
    def _simulate_instagram_engagement(self, start_date, end_date):
        """
        Simulate Instagram engagement data for development purposes.
        
        Args:
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            list: List of engagement dictionaries
        """
        # Simulate API rate limiting
        time.sleep(0.1)
        
        # Generate sample data
        engagement_data = []
        current_date = start_date
        
        while current_date <= end_date:
            # Generate hourly engagement data
            for hour in range(24):
                # More engagement during peak hours (8am-10pm)
                engagement_multiplier = 1.0
                if 8 <= hour <= 22:
                    engagement_multiplier = 1.5 + (hour % 5) * 0.3
                
                time_point = current_date.replace(hour=hour)
                
                engagement_data.append({
                    "timestamp": time_point.isoformat(),
                    "active_users": int(200 * engagement_multiplier + (hour * 10) % 100),
                    "likes_received": int(50 * engagement_multiplier + (hour * 5) % 50),
                    "comments_received": int(10 * engagement_multiplier + (hour * 2) % 15),
                    "shares_received": int(5 * engagement_multiplier + hour % 10),
                    "story_views": int(300 * engagement_multiplier + (hour * 20) % 200),
                    "profile_clicks": int(30 * engagement_multiplier + (hour * 3) % 30)
                })
            
            # Move to next day
            current_date += timedelta(days=1)
        
        return engagement_data 