# src/data/preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def preprocess_data(df: pd.DataFrame, feature_cols: list) -> tuple[pd.DataFrame, object]:
    """
    Clean, filter and scale AIS data.
    Returns processed dataframe and fitted scaler.
    """
    print("Preprocessing data...")

    # Basic cleaning
    df = df.dropna(subset=feature_cols).copy()

    # Remove invalid positions
    df = df[
        (df['lat'].between(-90, 90)) &
        (df['lon'].between(-180, 180)) &
        (df.get('sog', pd.Series(0)).between(0, 100))  # knots
    ]

    # Sort by timestamp and MMSI if available
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['mmsi', 'timestamp'] if 'mmsi' in df.columns else ['timestamp'])

    # Handle missing values (forward fill per vessel)
    if 'mmsi' in df.columns:
        df[feature_cols] = df.groupby('mmsi')[feature_cols].ffill()

    # Select only desired features
    df = df[feature_cols + (['mmsi', 'timestamp'] if 'mmsi' in df.columns else [])]

    # Scaling
    scaler = MinMaxScaler()  # or StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df[feature_cols]),
        columns=feature_cols,
        index=df.index
    )

    # Merge back metadata if needed
    if 'mmsi' in df.columns or 'timestamp' in df.columns:
        df_scaled = pd.concat([df[['mmsi', 'timestamp']], df_scaled], axis=1)

    print(f"Processed shape: {df_scaled.shape}")
    return df_scaled, scaler
