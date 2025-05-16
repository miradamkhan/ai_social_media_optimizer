import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from ..utils.config_loader import config
from ..utils.logger import get_logger

logger = get_logger(__name__)

class RetentionAnalysisModel:
    """
    Model for analyzing user retention and clustering users based on interaction behaviors
    """
    
    def __init__(self):
        """
        Initialize the RetentionAnalysisModel with configuration settings.
        """
        self.config = config.get_config("models").get("retention_analysis", {})
        self.algorithm = self.config.get("algorithm", "kmeans")
        self.params = self.config.get("params", {
            "n_clusters": 5,
            "random_state": 42
        })
        self.feature_cols = self.config.get("features", [
            "engagement_frequency",
            "time_since_last_interaction",
            "interaction_depth",
            "session_duration"
        ])
        
        # Initialize model and scaler
        self.model = None
        self.scaler = StandardScaler()
        self.pca = None
        self.cluster_profiles = {}
    
    def train(self, df):
        """
        Train the retention analysis model.
        
        Args:
            df (pandas.DataFrame): DataFrame with user engagement data
            
        Returns:
            dict: Training results
        """
        logger.info("Training retention analysis model")
        
        # Check if we have enough data
        if len(df) < 10:
            logger.error("Not enough data for clustering")
            return {"success": False, "error": "Not enough data"}
        
        # Check if required columns exist
        available_cols = [col for col in self.feature_cols if col in df.columns]
        if not available_cols:
            logger.error("No required columns available for training")
            return {"success": False, "error": "Missing required columns"}
        
        logger.info(f"Using features: {available_cols}")
        
        # Prepare data
        X = df[available_cols]
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale data
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA for visualization if we have more than 2 features
        if len(available_cols) > 2:
            self.pca = PCA(n_components=2)
            self.pca.fit(X_scaled)
        
        # Train model
        if self.algorithm == "kmeans":
            self.model = KMeans(
                n_clusters=self.params.get("n_clusters", 5),
                random_state=self.params.get("random_state", 42),
                n_init=10
            )
            
            # Fit model
            self.model.fit(X_scaled)
            
            # Get cluster labels
            labels = self.model.labels_
            
            # Add cluster labels to original data
            df_with_clusters = df.copy()
            df_with_clusters["cluster"] = labels
            
            # Analyze clusters
            self._analyze_clusters(df_with_clusters, available_cols)
            
            # Calculate metrics
            inertia = self.model.inertia_
            
            logger.info(f"Model trained with inertia: {inertia:.4f}")
            
            return {
                "success": True,
                "metrics": {
                    "inertia": inertia,
                    "n_clusters": self.params.get("n_clusters", 5)
                },
                "cluster_profiles": self.cluster_profiles
            }
        else:
            logger.error(f"Unsupported algorithm: {self.algorithm}")
            return {"success": False, "error": f"Unsupported algorithm: {self.algorithm}"}
    
    def _analyze_clusters(self, df, feature_cols):
        """
        Analyze the characteristics of each cluster.
        
        Args:
            df (pandas.DataFrame): DataFrame with cluster labels
            feature_cols (list): List of feature columns used for clustering
        """
        logger.info("Analyzing clusters")
        
        # Get number of clusters
        n_clusters = df["cluster"].nunique()
        
        # Analyze each cluster
        for cluster in range(n_clusters):
            cluster_data = df[df["cluster"] == cluster]
            
            # Calculate cluster metrics
            cluster_size = len(cluster_data)
            cluster_pct = cluster_size / len(df) * 100
            
            # Calculate mean values for each feature
            feature_means = {}
            for col in feature_cols:
                feature_means[col] = cluster_data[col].mean()
            
            # Calculate retention metrics
            retention_rate = None
            if "retention_score" in df.columns:
                retention_rate = cluster_data["retention_score"].mean()
            
            engagement_rate = None
            if "engagement_per_follower" in df.columns:
                engagement_rate = cluster_data["engagement_per_follower"].mean()
            
            # Store cluster profile
            self.cluster_profiles[cluster] = {
                "size": cluster_size,
                "percentage": cluster_pct,
                "feature_means": feature_means,
                "retention_rate": retention_rate,
                "engagement_rate": engagement_rate
            }
        
        # Assign descriptive labels to clusters
        self._assign_cluster_labels()
        
        logger.info(f"Analyzed {n_clusters} clusters")
    
    def _assign_cluster_labels(self):
        """
        Assign descriptive labels to clusters based on their characteristics.
        """
        # Define cluster labels based on engagement and retention metrics
        for cluster, profile in self.cluster_profiles.items():
            feature_means = profile["feature_means"]
            
            # Initialize score components
            engagement_score = 0
            recency_score = 0
            depth_score = 0
            
            # Calculate scores based on available features
            if "engagement_frequency" in feature_means:
                engagement_score = feature_means["engagement_frequency"]
                
            if "time_since_last_interaction" in feature_means:
                # Lower time since last interaction is better
                recency_score = -feature_means["time_since_last_interaction"]
                
            if "interaction_depth" in feature_means:
                depth_score = feature_means["interaction_depth"]
            
            # Combine scores
            total_score = engagement_score + recency_score + depth_score
            
            # Assign label based on total score
            if total_score > 1.5:
                label = "Highly Engaged"
            elif total_score > 0.5:
                label = "Regular User"
            elif total_score > -0.5:
                label = "Casual User"
            elif total_score > -1.5:
                label = "At Risk"
            else:
                label = "Disengaged"
            
            # Add label to profile
            self.cluster_profiles[cluster]["label"] = label
    
    def analyze_retention(self, df):
        """
        Analyze user retention by clustering new data.
        
        Args:
            df (pandas.DataFrame): DataFrame with user engagement data
            
        Returns:
            dict: Dictionary with retention analysis results
        """
        if not self.model:
            logger.warning("Model not trained yet")
            return {}
        
        logger.info("Analyzing user retention")
        
        # Check if required columns exist
        available_cols = [col for col in self.feature_cols if col in df.columns]
        if not available_cols:
            logger.error("No required columns available for analysis")
            return {}
        
        # Prepare data
        X = df[available_cols]
        X = X.fillna(X.mean())
        
        # Scale data
        X_scaled = self.scaler.transform(X)
        
        # Predict clusters
        labels = self.model.predict(X_scaled)
        
        # Add cluster labels to data
        df_with_clusters = df.copy()
        df_with_clusters["cluster"] = labels
        
        # Count users in each cluster
        cluster_counts = df_with_clusters["cluster"].value_counts().to_dict()
        
        # Calculate cluster percentages
        total_users = len(df)
        cluster_percentages = {
            cluster: count / total_users * 100
            for cluster, count in cluster_counts.items()
        }
        
        # Create result dictionary
        result = {
            "cluster_counts": cluster_counts,
            "cluster_percentages": cluster_percentages,
            "cluster_profiles": self.cluster_profiles,
            "total_users": total_users
        }
        
        # Add PCA visualization data if available
        if self.pca is not None:
            pca_result = self.pca.transform(X_scaled)
            
            # Convert to list for easier serialization
            visualization_data = []
            for i, (x, y) in enumerate(pca_result):
                visualization_data.append({
                    "x": float(x),
                    "y": float(y),
                    "cluster": int(labels[i])
                })
                
            result["visualization_data"] = visualization_data
        
        return result 