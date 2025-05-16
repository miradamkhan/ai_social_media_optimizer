import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_posts_data(start_date, days=30, posts_per_day=1):
    """
    Generate sample posts data.
    
    Args:
        start_date (datetime): Start date for data generation
        days (int): Number of days to generate data for
        posts_per_day (int): Average number of posts per day
        
    Returns:
        pandas.DataFrame: Generated posts data
    """
    # Initialize lists for data
    post_ids = []
    timestamps = []
    content_types = []
    caption_lengths = []
    hashtags_counts = []
    mentions_counts = []
    likes_counts = []
    comments_counts = []
    shares_counts = []
    saves_counts = []
    reach_counts = []
    impressions_counts = []
    
    # Generate data
    post_count = 1
    for day in range(days):
        current_date = start_date + timedelta(days=day)
        
        # Random number of posts for this day (1-3)
        daily_posts = np.random.randint(1, 4)
        
        for _ in range(daily_posts):
            # Random hour (weighted towards business hours)
            hour = np.random.choice(
                np.arange(24),
                p=np.array([
                    0.01, 0.01, 0.01, 0.01, 0.01, 0.02,  # 0-5
                    0.03, 0.05, 0.07, 0.08, 0.09, 0.09,  # 6-11
                    0.09, 0.08, 0.07, 0.07, 0.06, 0.05,  # 12-17
                    0.04, 0.04, 0.03, 0.02, 0.02, 0.01   # 18-23
                ])
            )
            
            # Create timestamp
            timestamp = current_date.replace(hour=hour, minute=np.random.randint(0, 60))
            
            # Content type
            content_type = np.random.choice(["image", "video", "carousel"], p=[0.5, 0.3, 0.2])
            
            # Caption length
            caption_length = np.random.randint(50, 200)
            
            # Hashtags and mentions
            hashtags = np.random.randint(2, 10)
            mentions = np.random.randint(0, 4)
            
            # Engagement metrics
            base_likes = {
                "image": np.random.normal(400, 50),
                "video": np.random.normal(800, 100),
                "carousel": np.random.normal(600, 75)
            }[content_type]
            
            likes = max(10, int(base_likes * np.random.normal(1, 0.1)))
            comments = max(1, int(likes * np.random.normal(0.08, 0.02)))
            shares = max(0, int(likes * np.random.normal(0.05, 0.01)))
            saves = max(0, int(likes * np.random.normal(0.03, 0.01)))
            
            # Reach and impressions
            reach = max(100, int(likes * np.random.normal(5, 1)))
            impressions = max(reach, int(reach * np.random.normal(1.2, 0.1)))
            
            # Append to lists
            post_ids.append(f"post_{post_count}")
            timestamps.append(timestamp)
            content_types.append(content_type)
            caption_lengths.append(caption_length)
            hashtags_counts.append(hashtags)
            mentions_counts.append(mentions)
            likes_counts.append(likes)
            comments_counts.append(comments)
            shares_counts.append(shares)
            saves_counts.append(saves)
            reach_counts.append(reach)
            impressions_counts.append(impressions)
            
            post_count += 1
    
    # Create DataFrame
    df = pd.DataFrame({
        "post_id": post_ids,
        "timestamp": timestamps,
        "content_type": content_types,
        "caption_length": caption_lengths,
        "hashtags": hashtags_counts,
        "mentions": mentions_counts,
        "likes": likes_counts,
        "comments": comments_counts,
        "shares": shares_counts,
        "saves": saves_counts,
        "reach": reach_counts,
        "impressions": impressions_counts
    })
    
    # Sort by timestamp
    df = df.sort_values("timestamp")
    
    return df

def generate_followers_data(start_date, days=30, initial_followers=10000):
    """
    Generate sample followers data.
    
    Args:
        start_date (datetime): Start date for data generation
        days (int): Number of days to generate data for
        initial_followers (int): Initial follower count
        
    Returns:
        pandas.DataFrame: Generated followers data
    """
    # Initialize lists for data
    dates = []
    followers_counts = []
    followers_gained = []
    followers_lost = []
    profile_views = []
    reach_values = []
    
    # Generate data
    followers = initial_followers
    
    for day in range(days):
        current_date = start_date + timedelta(days=day)
        
        # Followers gained and lost
        gained = max(10, int(np.random.normal(50 + day, 10)))
        lost = max(0, int(np.random.normal(10 + day * 0.5, 5)))
        
        # Update followers count
        followers += gained - lost
        
        # Profile views and reach
        views = max(100, int(followers * np.random.normal(0.05, 0.01)))
        reach = max(500, int(views * np.random.normal(5, 1)))
        
        # Append to lists
        dates.append(current_date)
        followers_counts.append(followers)
        followers_gained.append(gained)
        followers_lost.append(lost)
        profile_views.append(views)
        reach_values.append(reach)
    
    # Create DataFrame
    df = pd.DataFrame({
        "date": dates,
        "followers_count": followers_counts,
        "followers_gained": followers_gained,
        "followers_lost": followers_lost,
        "profile_views": profile_views,
        "reach": reach_values
    })
    
    return df

def generate_engagement_data(start_date, days=30):
    """
    Generate sample engagement data.
    
    Args:
        start_date (datetime): Start date for data generation
        days (int): Number of days to generate data for
        
    Returns:
        pandas.DataFrame: Generated engagement data
    """
    # Initialize lists for data
    timestamps = []
    active_users = []
    likes_received = []
    comments_received = []
    shares_received = []
    story_views = []
    profile_clicks = []
    
    # Generate data
    for day in range(days):
        current_date = start_date + timedelta(days=day)
        
        for hour in range(24):
            # Create timestamp
            timestamp = current_date.replace(hour=hour, minute=0)
            
            # Time-based activity factor (higher during day, lower at night)
            if 8 <= hour <= 22:
                activity_factor = 0.5 + 0.5 * np.sin(np.pi * (hour - 8) / 14)
            else:
                activity_factor = 0.2
            
            # Random variation
            daily_factor = np.random.normal(1, 0.1)
            
            # Active users
            base_users = 300
            users = max(10, int(base_users * activity_factor * daily_factor))
            
            # Engagement metrics
            likes = max(5, int(users * np.random.normal(0.25, 0.05)))
            comments = max(1, int(likes * np.random.normal(0.2, 0.05)))
            shares = max(0, int(comments * np.random.normal(0.5, 0.1)))
            views = max(50, int(users * np.random.normal(1.5, 0.2)))
            clicks = max(5, int(users * np.random.normal(0.1, 0.02)))
            
            # Append to lists
            timestamps.append(timestamp)
            active_users.append(users)
            likes_received.append(likes)
            comments_received.append(comments)
            shares_received.append(shares)
            story_views.append(views)
            profile_clicks.append(clicks)
    
    # Create DataFrame
    df = pd.DataFrame({
        "timestamp": timestamps,
        "active_users": active_users,
        "likes_received": likes_received,
        "comments_received": comments_received,
        "shares_received": shares_received,
        "story_views": story_views,
        "profile_clicks": profile_clicks
    })
    
    # Sort by timestamp
    df = df.sort_values("timestamp")
    
    return df

def generate_all_sample_data(output_dir="data/sample", days=30):
    """
    Generate all sample data files.
    
    Args:
        output_dir (str): Output directory
        days (int): Number of days to generate data for
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Start date (30 days ago)
    start_date = datetime.now() - timedelta(days=days)
    start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Generate data
    posts_df = generate_posts_data(start_date, days)
    followers_df = generate_followers_data(start_date, days)
    engagement_df = generate_engagement_data(start_date, days)
    
    # Save to CSV
    posts_df.to_csv(os.path.join(output_dir, "posts.csv"), index=False)
    followers_df.to_csv(os.path.join(output_dir, "followers.csv"), index=False)
    engagement_df.to_csv(os.path.join(output_dir, "engagement.csv"), index=False)
    
    print(f"Generated sample data in {output_dir}:")
    print(f"  - Posts: {len(posts_df)} rows")
    print(f"  - Followers: {len(followers_df)} rows")
    print(f"  - Engagement: {len(engagement_df)} rows")


if __name__ == "__main__":
    generate_all_sample_data() 