import os
import logging
from logging.handlers import RotatingFileHandler
from .config_loader import config

class Logger:
    """
    Utility class for setting up and managing application logging.
    """
    
    def __init__(self):
        """
        Initialize the Logger with configuration from the config file.
        """
        log_config = config.get_config("logging")
        self.log_level = log_config.get("level", "INFO")
        self.log_file = log_config.get("log_file", "logs/app.log")
        self.rotation = log_config.get("rotation", True)
        self.max_size_mb = log_config.get("max_size_mb", 10)
        self.backup_count = log_config.get("backup_count", 5)
        
        # Ensure log directory exists
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        # Set up the logger
        self._setup_logger()
    
    def _setup_logger(self):
        """
        Configure the logger with the specified settings.
        """
        # Convert string log level to logging constant
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        log_level = level_map.get(self.log_level.upper(), logging.INFO)
        
        # Configure root logger
        logger = logging.getLogger()
        logger.setLevel(log_level)
        
        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create formatters and handlers
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler (with rotation if enabled)
        if self.rotation:
            file_handler = RotatingFileHandler(
                self.log_file,
                maxBytes=self.max_size_mb * 1024 * 1024,
                backupCount=self.backup_count
            )
        else:
            file_handler = logging.FileHandler(self.log_file)
            
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Log initial message
        logger.info(f"Logging initialized at level: {self.log_level}")
    
    def get_logger(self, name=None):
        """
        Get a logger instance.
        
        Args:
            name (str, optional): Logger name
            
        Returns:
            logging.Logger: Logger instance
        """
        return logging.getLogger(name)


# Create singleton logger instance
app_logger = Logger()

# Export a function to get loggers
def get_logger(name=None):
    """
    Get a configured logger instance.
    
    Args:
        name (str, optional): Logger name
        
    Returns:
        logging.Logger: Logger instance
    """
    return app_logger.get_logger(name) 