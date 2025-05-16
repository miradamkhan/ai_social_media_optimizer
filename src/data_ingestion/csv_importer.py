import os
import pandas as pd
from datetime import datetime
from ..utils.config_loader import config
from ..utils.logger import get_logger

logger = get_logger(__name__)

class CSVImporter:
    """
    Class for importing data from CSV files
    """
    
    def __init__(self):
        """
        Initialize the CSV importer with configuration settings.
        """
        self.data_sources = config.get_config("data_sources")
    
    def import_data(self, platform, data_type=None, start_date=None, end_date=None):
        """
        Import data from CSV files for the specified platform.
        
        Args:
            platform (str): Social media platform to import data from
            data_type (str, optional): Type of data to import (posts, followers, engagement)
            start_date (str, optional): Start date for data filtering (YYYY-MM-DD)
            end_date (str, optional): End date for data filtering (YYYY-MM-DD)
            
        Returns:
            dict: Dictionary of DataFrames with keys for each data type
        """
        if platform not in self.data_sources:
            logger.error(f"Unsupported platform: {platform}")
            return {}
            
        # Get data path from configuration
        data_path = self.data_sources.get(platform, {}).get("data_path")
        if not data_path:
            logger.error(f"No data path configured for platform: {platform}")
            return {}
            
        # Check if data path exists
        if not os.path.exists(data_path):
            logger.warning(f"Data path does not exist: {data_path}")
            # Try to create the directory
            try:
                os.makedirs(data_path, exist_ok=True)
                logger.info(f"Created data directory: {data_path}")
            except Exception as e:
                logger.error(f"Failed to create data directory: {str(e)}")
                return {}
        
        result = {}
        
        # Import posts data if requested or if no specific type
        if data_type in [None, "posts"]:
            posts_file = os.path.join(data_path, "posts.csv")
            if os.path.exists(posts_file):
                try:
                    posts_df = self._load_and_filter_csv(posts_file, "timestamp", start_date, end_date)
                    result["posts"] = posts_df
                except Exception as e:
                    logger.error(f"Error loading posts data: {str(e)}")
            else:
                logger.warning(f"Posts file not found: {posts_file}")
        
        # Import followers data if requested or if no specific type
        if data_type in [None, "followers"]:
            followers_file = os.path.join(data_path, "followers.csv")
            if os.path.exists(followers_file):
                try:
                    followers_df = self._load_and_filter_csv(followers_file, "date", start_date, end_date)
                    result["followers"] = followers_df
                except Exception as e:
                    logger.error(f"Error loading followers data: {str(e)}")
            else:
                logger.warning(f"Followers file not found: {followers_file}")
        
        # Import engagement data if requested or if no specific type
        if data_type in [None, "engagement"]:
            engagement_file = os.path.join(data_path, "engagement.csv")
            if os.path.exists(engagement_file):
                try:
                    engagement_df = self._load_and_filter_csv(engagement_file, "timestamp", start_date, end_date)
                    result["engagement"] = engagement_df
                except Exception as e:
                    logger.error(f"Error loading engagement data: {str(e)}")
            else:
                logger.warning(f"Engagement file not found: {engagement_file}")
        
        return result
    
    def _load_and_filter_csv(self, file_path, date_column, start_date=None, end_date=None):
        """
        Load a CSV file and filter by date range if specified.
        
        Args:
            file_path (str): Path to the CSV file
            date_column (str): Name of the date column
            start_date (str, optional): Start date for filtering (YYYY-MM-DD)
            end_date (str, optional): End date for filtering (YYYY-MM-DD)
            
        Returns:
            pandas.DataFrame: Loaded and filtered DataFrame
        """
        logger.info(f"Loading CSV file: {file_path}")
        
        # Load the CSV file
        df = pd.read_csv(file_path)
        
        # Check if date column exists
        if date_column not in df.columns:
            logger.warning(f"Date column '{date_column}' not found in {file_path}")
            return df
        
        # Convert date column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            try:
                df[date_column] = pd.to_datetime(df[date_column])
            except Exception as e:
                logger.error(f"Error converting date column: {str(e)}")
                return df
        
        # Filter by date range if specified
        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            df = df[df[date_column] >= start_dt]
            
        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            df = df[df[date_column] <= end_dt]
        
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        return df
    
    def export_data(self, data_dict, platform):
        """
        Export data to CSV files.
        
        Args:
            data_dict (dict): Dictionary of DataFrames to export
            platform (str): Social media platform
            
        Returns:
            bool: True if export was successful
        """
        if platform not in self.data_sources:
            logger.error(f"Unsupported platform: {platform}")
            return False
            
        # Get data path from configuration
        data_path = self.data_sources.get(platform, {}).get("data_path")
        if not data_path:
            logger.error(f"No data path configured for platform: {platform}")
            return False
            
        # Ensure data directory exists
        os.makedirs(data_path, exist_ok=True)
        
        success = True
        
        # Export each DataFrame to a CSV file
        for data_type, df in data_dict.items():
            file_path = os.path.join(data_path, f"{data_type}.csv")
            try:
                df.to_csv(file_path, index=False)
                logger.info(f"Exported {len(df)} rows to {file_path}")
            except Exception as e:
                logger.error(f"Error exporting {data_type} data: {str(e)}")
                success = False
        
        return success 