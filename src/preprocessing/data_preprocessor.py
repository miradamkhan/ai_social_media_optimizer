import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from ..utils.config_loader import config
from ..utils.logger import get_logger

logger = get_logger(__name__)

class DataPreprocessor:
    """
    Class for preprocessing social media data
    """
    
    def __init__(self):
        """
        Initialize the DataPreprocessor with configuration settings.
        """
        self.config = config.get_config("preprocessing")
        self.normalize_metrics = self.config.get("normalize_metrics", True)
        self.remove_outliers = self.config.get("remove_outliers", True)
        self.outlier_threshold = self.config.get("outlier_threshold", 3.0)
        self.fill_missing_strategy = self.config.get("fill_missing_strategy", "mean")
        self.feature_scaling = self.config.get("feature_scaling", "standard")
        
        # Initialize scalers
        self.scalers = {}
    
    def preprocess(self, data_dict):
        """
        Preprocess the data dictionary containing DataFrames.
        
        Args:
            data_dict (dict): Dictionary of DataFrames with keys for each data type
            
        Returns:
            dict: Dictionary of preprocessed DataFrames
        """
        logger.info("Starting data preprocessing")
        
        result = {}
        
        for data_type, df in data_dict.items():
            logger.info(f"Preprocessing {data_type} data")
            
            # Make a copy to avoid modifying the original
            processed_df = df.copy()
            
            # Handle missing values
            processed_df = self._handle_missing_values(processed_df)
            
            # Remove outliers if enabled
            if self.remove_outliers:
                processed_df = self._remove_outliers(processed_df)
            
            # Normalize metrics if enabled
            if self.normalize_metrics:
                processed_df = self._normalize_metrics(processed_df, data_type)
            
            # Add to result
            result[data_type] = processed_df
            
            logger.info(f"Finished preprocessing {data_type} data: {len(processed_df)} rows")
        
        return result
    
    def _handle_missing_values(self, df):
        """
        Handle missing values in the DataFrame.
        
        Args:
            df (pandas.DataFrame): Input DataFrame
            
        Returns:
            pandas.DataFrame: DataFrame with missing values handled
        """
        # Get columns with missing values
        missing_cols = df.columns[df.isna().any()].tolist()
        
        if not missing_cols:
            return df
            
        logger.info(f"Handling missing values in columns: {missing_cols}")
        
        for col in missing_cols:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                logger.info(f"Column {col} has {missing_count} missing values")
                
                # Apply the appropriate fill strategy
                if self.fill_missing_strategy == "mean":
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].fillna(df[col].mean())
                    else:
                        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "")
                elif self.fill_missing_strategy == "median":
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].fillna(df[col].median())
                    else:
                        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "")
                elif self.fill_missing_strategy == "zero":
                    df[col] = df[col].fillna(0 if pd.api.types.is_numeric_dtype(df[col]) else "")
                elif self.fill_missing_strategy == "forward":
                    df[col] = df[col].fillna(method="ffill")
                elif self.fill_missing_strategy == "backward":
                    df[col] = df[col].fillna(method="bfill")
        
        return df
    
    def _remove_outliers(self, df):
        """
        Remove outliers from the DataFrame.
        
        Args:
            df (pandas.DataFrame): Input DataFrame
            
        Returns:
            pandas.DataFrame: DataFrame with outliers removed
        """
        original_len = len(df)
        
        # Only apply to numeric columns
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        if not numeric_cols:
            return df
            
        logger.info(f"Removing outliers from numeric columns: {numeric_cols}")
        
        # Calculate z-scores for numeric columns
        for col in numeric_cols:
            col_mean = df[col].mean()
            col_std = df[col].std()
            
            if col_std == 0:
                continue
                
            z_scores = np.abs((df[col] - col_mean) / col_std)
            df = df[z_scores < self.outlier_threshold]
        
        removed_count = original_len - len(df)
        logger.info(f"Removed {removed_count} outliers ({removed_count/original_len:.2%} of data)")
        
        return df
    
    def _normalize_metrics(self, df, data_type):
        """
        Normalize numeric metrics in the DataFrame.
        
        Args:
            df (pandas.DataFrame): Input DataFrame
            data_type (str): Type of data being normalized
            
        Returns:
            pandas.DataFrame: DataFrame with normalized metrics
        """
        # Only normalize numeric columns
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        if not numeric_cols:
            return df
            
        logger.info(f"Normalizing metrics: {numeric_cols}")
        
        # Create a scaler based on the configuration
        if data_type not in self.scalers:
            if self.feature_scaling == "standard":
                self.scalers[data_type] = StandardScaler()
            elif self.feature_scaling == "minmax":
                self.scalers[data_type] = MinMaxScaler()
            elif self.feature_scaling == "robust":
                self.scalers[data_type] = RobustScaler()
            else:
                logger.warning(f"Unknown scaling method: {self.feature_scaling}, using StandardScaler")
                self.scalers[data_type] = StandardScaler()
        
        # Apply scaling
        scaler = self.scalers[data_type]
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        return df
    
    def transform_new_data(self, df, data_type):
        """
        Transform new data using the fitted preprocessor.
        
        Args:
            df (pandas.DataFrame): Input DataFrame
            data_type (str): Type of data being transformed
            
        Returns:
            pandas.DataFrame: Transformed DataFrame
        """
        logger.info(f"Transforming new {data_type} data")
        
        # Make a copy to avoid modifying the original
        transformed_df = df.copy()
        
        # Handle missing values
        transformed_df = self._handle_missing_values(transformed_df)
        
        # Apply scaling if a scaler exists for this data type
        if data_type in self.scalers:
            numeric_cols = transformed_df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                transformed_df[numeric_cols] = self.scalers[data_type].transform(transformed_df[numeric_cols])
        
        return transformed_df 