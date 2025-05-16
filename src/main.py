import os
import argparse
import pandas as pd
from src.data_ingestion import DataLoader
from src.preprocessing import DataPreprocessor, FeatureEngineer
from src.models import ModelManager
from src.dashboard import create_dashboard
from src.utils.config_loader import config
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    """
    Main entry point for the Social Media Analytics Tool.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Social Media Analytics Tool")
    parser.add_argument("--platform", help="Social media platform to analyze")
    parser.add_argument("--start-date", help="Start date for data (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date for data (YYYY-MM-DD)")
    parser.add_argument("--use-sample", action="store_true", help="Use sample data")
    parser.add_argument("--dashboard", action="store_true", help="Launch dashboard")
    parser.add_argument("--train", action="store_true", help="Train models")
    parser.add_argument("--optimize", action="store_true", help="Get optimization recommendations")
    args = parser.parse_args()
    
    logger.info("Starting Social Media Analytics Tool")
    
    # Initialize components
    data_loader = DataLoader()
    preprocessor = DataPreprocessor()
    feature_engineer = FeatureEngineer()
    model_manager = ModelManager()
    
    # Load data
    data = {}
    if args.use_sample:
        logger.info("Loading sample data")
        data = data_loader.get_sample_data()
    else:
        logger.info(f"Loading data for platform: {args.platform}")
        data = data_loader.load_data(
            platform=args.platform,
            start_date=args.start_date,
            end_date=args.end_date
        )
    
    if not data:
        logger.error("No data loaded. Exiting.")
        return
    
    # Preprocess data
    logger.info("Preprocessing data")
    processed_data = preprocessor.preprocess(data)
    
    # Engineer features
    logger.info("Engineering features")
    engineered_data = feature_engineer.engineer_features(processed_data)
    
    # Train models if requested
    if args.train:
        logger.info("Training models")
        training_results = model_manager.train_all_models(engineered_data)
        
        # Save trained models
        logger.info("Saving models")
        model_manager.save_models()
    else:
        # Try to load existing models
        logger.info("Loading existing models")
        model_manager.load_models()
    
    # Generate optimization recommendations if requested
    if args.optimize:
        logger.info("Generating optimization recommendations")
        
        # Get optimal posting times
        if model_manager.trained_models["posting_time"]:
            optimal_times = model_manager.get_optimal_posting_times()
            print("\nOptimal Posting Times:")
            for date, times in optimal_times.items():
                formatted_times = []
                for time in times:
                    if isinstance(time, str):
                        # If it's already a string, extract the hour and minute
                        formatted_times.append(':'.join(time.split()[1].split(':')[:2]))
                    else:
                        # If it's a list, just join the items
                        formatted_times.append(f"{time}:00")
                print(f"  {date}: {', '.join(formatted_times)}")
        
        # Get content recommendations
        if model_manager.trained_models["content_engagement"]:
            content_recommendations = model_manager.get_content_recommendations()
            print("\nContent Recommendations:")
            if "top_content_types" in content_recommendations:
                print(f"  Top Content Types: {', '.join(content_recommendations['top_content_types'])}")
        
        # Get follower growth forecast
        if model_manager.trained_models["follower_growth"]:
            growth_forecast = model_manager.predict_follower_growth()
            print("\nFollower Growth Forecast (next 7 days):")
            for i, (date, count) in enumerate(list(growth_forecast.items())[:7]):
                print(f"  {date}: {count} followers")
    
    # Launch dashboard if requested
    if args.dashboard:
        logger.info("Launching dashboard")
        app = create_dashboard(model_manager)
        
        # Get dashboard configuration
        dashboard_config = config.get_config("dashboard")
        
        # Run server
        app.run_server(
            debug=dashboard_config.get("debug", True),
            port=dashboard_config.get("port", 8050)
        )


if __name__ == "__main__":
    main() 