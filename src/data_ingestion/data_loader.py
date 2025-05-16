import pandas as pd
from ..utils.config_loader import config
from ..utils.logger import get_logger
from .api_client import APIClient
from .csv_importer import CSVImporter

logger = get_logger(__name__)

class DataLoader:
    """
    Main class for loading data from various sources (API or CSV files)
    """
    
    def __init__(self):
        """
        Initialize the DataLoader with configuration settings.
        """
        self.config = config.get_config("data_ingestion")
        self.data_sources = config.get_config("data_sources")
        self.use_api = self.config.get("use_api", False)
        self.use_csv = self.config.get("use_csv", True)
        
        # Initialize data clients
        if self.use_api:
            self.api_client = APIClient()
        
        if self.use_csv:
            self.csv_importer = CSVImporter()
    
    def load_data(self, platform=None, data_type=None, start_date=None, end_date=None):
        """
        Load data from the configured sources.
        
        Args:
            platform (str, optional): Social media platform to load data from
            data_type (str, optional): Type of data to load (posts, followers, engagement)
            start_date (str, optional): Start date for data filtering (YYYY-MM-DD)
            end_date (str, optional): End date for data filtering (YYYY-MM-DD)
            
        Returns:
            dict: Dictionary of DataFrames with keys for each data type
        """
        logger.info(f"Loading data for platform: {platform}, type: {data_type}")
        
        # Determine which platforms to load
        platforms = [platform] if platform else [p for p, cfg in self.data_sources.items() if cfg.get("enabled")]
        
        result = {}
        
        for platform in platforms:
            if not self.data_sources.get(platform, {}).get("enabled", False):
                logger.warning(f"Platform {platform} is disabled in configuration")
                continue
                
            platform_data = {}
            
            # Try loading from API first if enabled
            if self.use_api:
                try:
                    logger.info(f"Attempting to load {platform} data from API")
                    platform_data = self.api_client.fetch_data(
                        platform=platform,
                        data_type=data_type,
                        start_date=start_date,
                        end_date=end_date
                    )
                except Exception as e:
                    logger.error(f"Error loading {platform} data from API: {str(e)}")
            
            # If API failed or disabled, try CSV
            if not platform_data and self.use_csv:
                try:
                    logger.info(f"Loading {platform} data from CSV files")
                    platform_data = self.csv_importer.import_data(
                        platform=platform,
                        data_type=data_type,
                        start_date=start_date,
                        end_date=end_date
                    )
                except Exception as e:
                    logger.error(f"Error loading {platform} data from CSV: {str(e)}")
            
            # Add to results if data was loaded
            if platform_data:
                result[platform] = platform_data
            else:
                logger.warning(f"No data loaded for platform: {platform}")
        
        return result
    
    def get_sample_data(self):
        """
        Load sample data for testing and development.
        
        Returns:
            dict: Dictionary of sample DataFrames
        """
        logger.info("Loading sample data")
        sample_path = self.config.get("sample_data_path", "data/sample")
        
        try:
            posts_df = pd.read_csv(f"{sample_path}/posts.csv")
            followers_df = pd.read_csv(f"{sample_path}/followers.csv")
            engagement_df = pd.read_csv(f"{sample_path}/engagement.csv")
            
            return {
                "posts": posts_df,
                "followers": followers_df,
                "engagement": engagement_df
            }
        except Exception as e:
            logger.error(f"Error loading sample data: {str(e)}")
            return {} 