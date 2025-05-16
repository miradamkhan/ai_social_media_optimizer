#!/usr/bin/env python3
import os
import sys
import argparse
from src.main import main

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Social Media Analytics Tool")
    parser.add_argument("--platform", help="Social media platform to analyze (instagram, tiktok, twitter)")
    parser.add_argument("--start-date", help="Start date for data (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date for data (YYYY-MM-DD)")
    parser.add_argument("--use-sample", action="store_true", help="Use sample data")
    parser.add_argument("--dashboard", action="store_true", help="Launch dashboard")
    parser.add_argument("--train", action="store_true", help="Train models")
    parser.add_argument("--optimize", action="store_true", help="Get optimization recommendations")
    parser.add_argument("--generate-sample", action="store_true", help="Generate new sample data")
    
    args = parser.parse_args()
    
    # Generate sample data if requested
    if args.generate_sample:
        print("Generating sample data...")
        from src.utils.generate_sample_data import generate_all_sample_data
        generate_all_sample_data()
        print("Sample data generation complete.")
        
        if not any([args.platform, args.use_sample, args.dashboard, args.train, args.optimize]):
            sys.exit(0)
    
    # Set default values
    if not args.platform and not args.use_sample:
        print("No platform specified. Using sample data.")
        args.use_sample = True
    
    if not any([args.dashboard, args.train, args.optimize]):
        print("No action specified. Launching dashboard.")
        args.dashboard = True
    
    # Run main with parsed arguments
    sys.argv = [sys.argv[0]]  # Clear sys.argv to avoid conflicts with main's argparse
    
    # Pass arguments to main
    main_args = []
    
    if args.platform:
        main_args.extend(["--platform", args.platform])
    
    if args.start_date:
        main_args.extend(["--start-date", args.start_date])
    
    if args.end_date:
        main_args.extend(["--end-date", args.end_date])
    
    if args.use_sample:
        main_args.append("--use-sample")
    
    if args.dashboard:
        main_args.append("--dashboard")
    
    if args.train:
        main_args.append("--train")
    
    if args.optimize:
        main_args.append("--optimize")
    
    # Set sys.argv for main
    sys.argv.extend(main_args)
    
    # Run main
    main() 