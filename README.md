# Social Media Analytics AI Tool

An AI-powered tool for social media brands to maximize user retention, follower growth, and viewership.

## Features

- **Data Ingestion**: Import data from multiple social media platforms (Instagram, TikTok, Twitter) via APIs or CSV files
- **Data Preprocessing**: Clean, normalize, and extract relevant features from social media data
- **Machine Learning Models**:
  - Optimal posting time prediction using XGBoost
  - Content type engagement analysis with Random Forest
  - Follower growth forecasting using LSTM time series analysis
  - User retention clustering with K-means
- **Interactive Dashboard**: Visualize performance insights and growth forecasts
- **Optimization Recommendations**: Get actionable insights to improve social media performance

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Using the Command-Line Interface

```bash
# Generate sample data
python run.py --generate-sample

# Launch the dashboard with sample data
python run.py --use-sample --dashboard

# Train models using sample data
python run.py --use-sample --train

# Get optimization recommendations
python run.py --use-sample --optimize

# Analyze a specific platform with date range
python run.py --platform instagram --start-date 2023-06-01 --end-date 2023-06-30 --dashboard
```

### Using the Dashboard

The dashboard provides visualizations for:
- Follower growth and forecasts
- Engagement metrics
- Optimal posting times
- Content performance analysis
- User retention clusters

Access the dashboard by running:
```
python run.py --dashboard
```

## Data Format

The system accepts CSV files with the following structure:
- `posts.csv`: Post metrics data
- `followers.csv`: Follower growth data
- `engagement.csv`: User engagement data

Example data files are provided in the `data/sample` directory.

## Configuration

Modify `config/config.yaml` to customize:
- Data sources
- Model parameters
- Preprocessing settings
- Dashboard configuration
- Logging levels

## Project Structure

```
.
├── config/                 # Configuration files
├── data/                   # Data directory
│   └── sample/             # Sample data files
├── logs/                   # Log files
├── src/                    # Source code
│   ├── data_ingestion/     # Data ingestion modules
│   ├── preprocessing/      # Data preprocessing modules
│   ├── models/             # Machine learning models
│   ├── dashboard/          # Dashboard application
│   └── utils/              # Utility functions
├── requirements.txt        # Dependencies
├── run.py                  # CLI entry point
└── README.md               # This file
```

## License

MIT License 
