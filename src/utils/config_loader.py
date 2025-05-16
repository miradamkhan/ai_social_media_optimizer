import os
import yaml
from dotenv import load_dotenv

class ConfigLoader:
    """
    Utility class for loading and accessing configuration settings from YAML files
    and environment variables.
    """
    
    def __init__(self, config_path="config/config.yaml"):
        """
        Initialize the ConfigLoader with the path to the configuration file.
        
        Args:
            config_path (str): Path to the YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_yaml_config()
        load_dotenv()  # Load environment variables from .env file
        
    def _load_yaml_config(self):
        """
        Load configuration from YAML file.
        
        Returns:
            dict: Configuration dictionary
        """
        try:
            with open(self.config_path, 'r') as config_file:
                return yaml.safe_load(config_file)
        except FileNotFoundError:
            print(f"Configuration file not found: {self.config_path}")
            return {}
        except yaml.YAMLError as e:
            print(f"Error parsing YAML configuration: {e}")
            return {}
    
    def get_config(self, section=None):
        """
        Get configuration settings, optionally filtered by section.
        
        Args:
            section (str, optional): Configuration section to retrieve
            
        Returns:
            dict: Configuration dictionary or section
        """
        if section is None:
            return self.config
        
        return self.config.get(section, {})
    
    def get_env_var(self, var_name, default=None):
        """
        Get environment variable value.
        
        Args:
            var_name (str): Name of the environment variable
            default: Default value if environment variable is not set
            
        Returns:
            str: Value of the environment variable or default
        """
        return os.environ.get(var_name, default)


# Singleton instance for global access
config = ConfigLoader() 