import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ..utils.config_loader import config
from ..utils.logger import get_logger

logger = get_logger(__name__)

def create_dashboard(model_manager):
    """
    Create the Dash application for visualizing insights.
    
    Args:
        model_manager: ModelManager instance with trained models
        
    Returns:
        dash.Dash: Dash application
    """
    logger.info("Creating dashboard")
    
    # Get dashboard configuration
    dashboard_config = config.get_config("dashboard")
    
    # Create Dash app
    app = dash.Dash(__name__, title="Social Media Analytics Dashboard")
    
    # Define layout
    app.layout = html.Div([
        html.H1("Social Media Analytics Dashboard", className="dashboard-title"),
        
        html.Div([
            html.Div([
                html.H3("Platform Selection"),
                dcc.Dropdown(
                    id="platform-dropdown",
                    options=[
                        {"label": "Instagram", "value": "instagram"},
                        {"label": "TikTok", "value": "tiktok"},
                        {"label": "Twitter", "value": "twitter"}
                    ],
                    value="instagram",
                    clearable=False
                )
            ], className="control-panel"),
            
            html.Div([
                html.H3("Date Range"),
                dcc.DatePickerRange(
                    id="date-range",
                    start_date=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
                    end_date=datetime.now().strftime("%Y-%m-%d"),
                    max_date_allowed=datetime.now().strftime("%Y-%m-%d")
                )
            ], className="control-panel")
        ], className="controls-row"),
        
        html.Div([
            html.Div([
                html.H3("Follower Growth"),
                dcc.Graph(id="follower-growth-chart")
            ], className="chart-container"),
            
            html.Div([
                html.H3("Engagement Metrics"),
                dcc.Graph(id="engagement-metrics-chart")
            ], className="chart-container")
        ], className="charts-row"),
        
        html.Div([
            html.Div([
                html.H3("Optimal Posting Times"),
                dcc.Graph(id="posting-times-chart")
            ], className="chart-container"),
            
            html.Div([
                html.H3("Content Performance"),
                dcc.Graph(id="content-performance-chart")
            ], className="chart-container")
        ], className="charts-row"),
        
        html.Div([
            html.H3("User Retention Analysis"),
            html.Div([
                html.Div([
                    dcc.Graph(id="user-clusters-chart")
                ], className="chart-container-half"),
                
                html.Div([
                    html.H4("Cluster Profiles"),
                    html.Div(id="cluster-profiles-table")
                ], className="chart-container-half")
            ], className="cluster-analysis-row")
        ], className="full-width-container"),
        
        html.Div([
            html.H3("Growth Forecast"),
            dcc.Graph(id="growth-forecast-chart")
        ], className="full-width-container"),
        
        # Store for intermediate data
        dcc.Store(id="follower-data-store"),
        dcc.Store(id="engagement-data-store"),
        dcc.Store(id="content-data-store"),
        dcc.Store(id="retention-data-store"),
        
        # Interval for auto-refresh
        dcc.Interval(
            id="refresh-interval",
            interval=dashboard_config.get("refresh_interval_seconds", 300) * 1000,  # in milliseconds
            n_intervals=0
        )
    ], className="dashboard-container")
    
    # Define callback to update follower growth chart
    @app.callback(
        Output("follower-growth-chart", "figure"),
        [Input("platform-dropdown", "value"),
         Input("date-range", "start_date"),
         Input("date-range", "end_date"),
         Input("follower-data-store", "data")]
    )
    def update_follower_growth_chart(platform, start_date, end_date, stored_data):
        """
        Update the follower growth chart based on selected platform and date range.
        """
        logger.info(f"Updating follower growth chart for {platform}")
        
        # Create sample data for demonstration
        dates = pd.date_range(start=start_date, end=end_date)
        followers = np.cumsum(np.random.normal(50, 10, size=len(dates))) + 10000
        
        # Create DataFrame
        df = pd.DataFrame({
            "date": dates,
            "followers": followers
        })
        
        # Create figure
        fig = px.line(
            df,
            x="date",
            y="followers",
            title=f"{platform.capitalize()} Follower Growth"
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Followers",
            template="plotly_white"
        )
        
        return fig
    
    # Define callback to update engagement metrics chart
    @app.callback(
        Output("engagement-metrics-chart", "figure"),
        [Input("platform-dropdown", "value"),
         Input("date-range", "start_date"),
         Input("date-range", "end_date"),
         Input("engagement-data-store", "data")]
    )
    def update_engagement_metrics_chart(platform, start_date, end_date, stored_data):
        """
        Update the engagement metrics chart based on selected platform and date range.
        """
        logger.info(f"Updating engagement metrics chart for {platform}")
        
        # Create sample data for demonstration
        dates = pd.date_range(start=start_date, end=end_date)
        likes = np.random.normal(500, 100, size=len(dates))
        comments = np.random.normal(50, 15, size=len(dates))
        shares = np.random.normal(25, 8, size=len(dates))
        
        # Create DataFrame
        df = pd.DataFrame({
            "date": dates,
            "likes": likes,
            "comments": comments,
            "shares": shares
        })
        
        # Create figure
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["likes"],
            name="Likes",
            line=dict(color="blue")
        ))
        
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["comments"],
            name="Comments",
            line=dict(color="green")
        ))
        
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df["shares"],
            name="Shares",
            line=dict(color="red")
        ))
        
        fig.update_layout(
            title=f"{platform.capitalize()} Engagement Metrics",
            xaxis_title="Date",
            yaxis_title="Count",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    # Define callback to update posting times chart
    @app.callback(
        Output("posting-times-chart", "figure"),
        [Input("platform-dropdown", "value"),
         Input("date-range", "start_date"),
         Input("date-range", "end_date")]
    )
    def update_posting_times_chart(platform, start_date, end_date):
        """
        Update the optimal posting times chart based on model predictions.
        """
        logger.info(f"Updating posting times chart for {platform}")
        
        # Get optimal posting times from model
        if model_manager.trained_models["posting_time"]:
            optimal_times = model_manager.get_optimal_posting_times(days=7)
        else:
            # Create sample data for demonstration
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            optimal_hours = {
                day: [9, 12, 17] for day in days
            }
            
            # Convert to expected format
            optimal_times = {}
            current_date = datetime.now()
            for i, day in enumerate(days):
                date_str = (current_date + timedelta(days=i)).strftime("%Y-%m-%d")
                optimal_times[date_str] = [
                    f"{date_str} {hour:02d}:00:00" for hour in optimal_hours[day]
                ]
        
        # Prepare data for heatmap
        hour_data = []
        day_data = []
        value_data = []
        
        for date_str, times in optimal_times.items():
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            day_name = date_obj.strftime("%A")
            
            # Extract hours from times
            hours = [int(time.split()[1].split(":")[0]) for time in times]
            
            # Create heatmap data
            for hour in range(24):
                hour_data.append(hour)
                day_data.append(day_name)
                
                # Value is 1 if hour is optimal, 0 otherwise
                value = 1 if hour in hours else 0
                value_data.append(value)
        
        # Create DataFrame
        df = pd.DataFrame({
            "hour": hour_data,
            "day": day_data,
            "value": value_data
        })
        
        # Create figure
        fig = px.density_heatmap(
            df,
            x="hour",
            y="day",
            z="value",
            title=f"Optimal Posting Times for {platform.capitalize()}",
            category_orders={"day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]}
        )
        
        fig.update_layout(
            xaxis_title="Hour of Day",
            yaxis_title="Day of Week",
            template="plotly_white"
        )
        
        return fig
    
    # Define callback to update content performance chart
    @app.callback(
        Output("content-performance-chart", "figure"),
        [Input("platform-dropdown", "value"),
         Input("content-data-store", "data")]
    )
    def update_content_performance_chart(platform, stored_data):
        """
        Update the content performance chart based on model recommendations.
        """
        logger.info(f"Updating content performance chart for {platform}")
        
        # Get content recommendations from model
        if model_manager.trained_models["content_engagement"]:
            recommendations = model_manager.get_content_recommendations()
            content_performance = recommendations.get("content_performance", {})
        else:
            # Create sample data for demonstration
            content_performance = {
                "image": 0.045,
                "video": 0.062,
                "carousel": 0.051,
                "text": 0.032,
                "link": 0.028
            }
        
        # Create DataFrame
        df = pd.DataFrame({
            "content_type": list(content_performance.keys()),
            "engagement_rate": list(content_performance.values())
        })
        
        # Sort by engagement rate
        df = df.sort_values("engagement_rate", ascending=False)
        
        # Create figure
        fig = px.bar(
            df,
            x="content_type",
            y="engagement_rate",
            title=f"Content Type Performance for {platform.capitalize()}",
            color="engagement_rate",
            color_continuous_scale="Viridis"
        )
        
        fig.update_layout(
            xaxis_title="Content Type",
            yaxis_title="Engagement Rate",
            template="plotly_white",
            coloraxis_showscale=False
        )
        
        return fig
    
    # Define callback to update user clusters chart
    @app.callback(
        Output("user-clusters-chart", "figure"),
        [Input("platform-dropdown", "value"),
         Input("retention-data-store", "data")]
    )
    def update_user_clusters_chart(platform, stored_data):
        """
        Update the user clusters chart based on retention analysis.
        """
        logger.info(f"Updating user clusters chart for {platform}")
        
        # Check if retention analysis model is trained and enabled
        if model_manager.trained_models["retention_analysis"] and model_manager.retention_analysis_model:
            # Get retention analysis results
            # In a real application, this would use actual data
            retention_data = pd.DataFrame({
                "engagement_frequency": np.random.normal(0.5, 0.2, 100),
                "time_since_last_interaction": np.random.normal(3, 1, 100),
                "interaction_depth": np.random.normal(0.4, 0.15, 100)
            })
            
            results = model_manager.analyze_user_retention(retention_data)
            visualization_data = results.get("visualization_data", [])
            
            if visualization_data:
                # Create DataFrame from visualization data
                viz_df = pd.DataFrame(visualization_data)
                
                # Get cluster profiles for labels
                cluster_profiles = results.get("cluster_profiles", {})
                
                # Create mapping from cluster ID to label
                cluster_labels = {
                    cluster_id: profile.get("label", f"Cluster {cluster_id}")
                    for cluster_id, profile in cluster_profiles.items()
                }
                
                # Add cluster labels
                viz_df["cluster_label"] = viz_df["cluster"].map(cluster_labels)
                
                # Create figure
                fig = px.scatter(
                    viz_df,
                    x="x",
                    y="y",
                    color="cluster_label",
                    title=f"User Clusters for {platform.capitalize()}",
                    labels={"x": "Component 1", "y": "Component 2", "cluster_label": "Cluster"}
                )
                
                fig.update_layout(
                    template="plotly_white",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                return fig
        
        # Create sample data for demonstration
        np.random.seed(42)
        n_clusters = 5
        
        # Generate cluster centers
        centers = np.random.uniform(-3, 3, size=(n_clusters, 2))
        
        # Generate points around centers
        points_per_cluster = 50
        x = []
        y = []
        cluster = []
        cluster_label = []
        
        for i in range(n_clusters):
            cluster_x = np.random.normal(centers[i, 0], 0.5, points_per_cluster)
            cluster_y = np.random.normal(centers[i, 1], 0.5, points_per_cluster)
            
            x.extend(cluster_x)
            y.extend(cluster_y)
            cluster.extend([i] * points_per_cluster)
            
            # Assign labels
            labels = ["Highly Engaged", "Regular User", "Casual User", "At Risk", "Disengaged"]
            cluster_label.extend([labels[i]] * points_per_cluster)
        
        # Create DataFrame
        df = pd.DataFrame({
            "x": x,
            "y": y,
            "cluster": cluster,
            "cluster_label": cluster_label
        })
        
        # Create figure
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="cluster_label",
            title=f"User Clusters for {platform.capitalize()}",
            labels={"x": "Component 1", "y": "Component 2", "cluster_label": "Cluster"}
        )
        
        fig.update_layout(
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    # Define callback to update cluster profiles table
    @app.callback(
        Output("cluster-profiles-table", "children"),
        [Input("platform-dropdown", "value"),
         Input("retention-data-store", "data")]
    )
    def update_cluster_profiles_table(platform, stored_data):
        """
        Update the cluster profiles table based on retention analysis.
        """
        logger.info(f"Updating cluster profiles table for {platform}")
        
        # Check if retention analysis model is trained and enabled
        if model_manager.trained_models["retention_analysis"] and model_manager.retention_analysis_model:
            # Get retention analysis results
            # In a real application, this would use actual data
            retention_data = pd.DataFrame({
                "engagement_frequency": np.random.normal(0.5, 0.2, 100),
                "time_since_last_interaction": np.random.normal(3, 1, 100),
                "interaction_depth": np.random.normal(0.4, 0.15, 100)
            })
            
            results = model_manager.analyze_user_retention(retention_data)
            cluster_profiles = results.get("cluster_profiles", {})
            
            if cluster_profiles:
                # Create table rows
                rows = []
                
                for cluster_id, profile in cluster_profiles.items():
                    label = profile.get("label", f"Cluster {cluster_id}")
                    size = profile.get("size", 0)
                    percentage = profile.get("percentage", 0)
                    
                    # Create row
                    row = html.Tr([
                        html.Td(label),
                        html.Td(f"{size} users"),
                        html.Td(f"{percentage:.1f}%")
                    ])
                    
                    rows.append(row)
                
                # Create table
                table = html.Table([
                    html.Thead(html.Tr([
                        html.Th("Cluster"),
                        html.Th("Size"),
                        html.Th("Percentage")
                    ])),
                    html.Tbody(rows)
                ], className="cluster-profiles-table")
                
                return table
        
        # Create sample data for demonstration
        labels = ["Highly Engaged", "Regular User", "Casual User", "At Risk", "Disengaged"]
        sizes = [120, 250, 350, 180, 100]
        total = sum(sizes)
        percentages = [size / total * 100 for size in sizes]
        
        # Create table rows
        rows = []
        
        for i in range(len(labels)):
            row = html.Tr([
                html.Td(labels[i]),
                html.Td(f"{sizes[i]} users"),
                html.Td(f"{percentages[i]:.1f}%")
            ])
            
            rows.append(row)
        
        # Create table
        table = html.Table([
            html.Thead(html.Tr([
                html.Th("Cluster"),
                html.Th("Size"),
                html.Th("Percentage")
            ])),
            html.Tbody(rows)
        ], className="cluster-profiles-table")
        
        return table
    
    # Define callback to update growth forecast chart
    @app.callback(
        Output("growth-forecast-chart", "figure"),
        [Input("platform-dropdown", "value")]
    )
    def update_growth_forecast_chart(platform):
        """
        Update the growth forecast chart based on model predictions.
        """
        logger.info(f"Updating growth forecast chart for {platform}")
        
        # Get growth forecast from model
        if model_manager.trained_models["follower_growth"]:
            forecast = model_manager.predict_follower_growth(days=30)
        else:
            # Create sample data for demonstration
            start_date = datetime.now()
            dates = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(30)]
            
            # Generate forecasted follower counts with some randomness
            base = 10000
            forecast = {}
            for i, date in enumerate(dates):
                forecast[date] = int(base + i * 60 + np.random.normal(0, 20))
        
        # Create DataFrame
        df = pd.DataFrame({
            "date": list(forecast.keys()),
            "followers": list(forecast.values())
        })
        
        # Convert date to datetime
        df["date"] = pd.to_datetime(df["date"])
        
        # Create figure
        fig = px.line(
            df,
            x="date",
            y="followers",
            title=f"{platform.capitalize()} Follower Growth Forecast (Next 30 Days)"
        )
        
        # Add confidence interval (for demonstration)
        upper = df["followers"] * 1.05
        lower = df["followers"] * 0.95
        
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=upper,
            fill=None,
            mode="lines",
            line_color="rgba(0,100,80,0)",
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=lower,
            fill="tonexty",
            mode="lines",
            line_color="rgba(0,100,80,0)",
            fillcolor="rgba(0,100,80,0.2)",
            name="95% Confidence Interval"
        ))
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Projected Followers",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    return app


if __name__ == "__main__":
    # This is for running the dashboard standalone
    from ..models.model_manager import ModelManager
    
    # Create model manager
    model_manager = ModelManager()
    
    # Create dashboard
    app = create_dashboard(model_manager)
    
    # Get dashboard configuration
    dashboard_config = config.get_config("dashboard")
    
    # Run server
    app.run_server(
        debug=dashboard_config.get("debug", True),
        port=dashboard_config.get("port", 8050)
    ) 