# src/utils/helpers.py
"""
Small helper functions used across the project
"""

import numpy as np
import math
from datetime import datetime


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great-circle distance between two points on Earth (km)
    """
    R = 6371.0  # Earth radius in kilometers

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance


def haversine_batch(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """
    Vectorized Haversine distance for numpy arrays (in km)
    Useful for evaluation metrics (ADE, FDE, etc.)
    """
    R = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def timestamp_to_unix(ts) -> int:
    """Convert pandas Timestamp / datetime to Unix timestamp (seconds)"""
    if isinstance(ts, (pd.Timestamp, datetime)):
        return int(ts.timestamp())
    return int(ts)


def get_vessel_type(mmsi: int) -> str:
    """
    Very simple MMSI-based vessel type estimation
    (real implementation should use proper lookup table or AIS type field)
    """
    if 100000000 <= mmsi < 200000000:
        return "Reserved"
    elif 200000000 <= mmsi < 800000000:
        return "Ship"
    elif 800000000 <= mmsi < 900000000:
        return "SAR Aircraft"
    elif 900000000 <= mmsi < 1000000000:
        return "AIS Aid to Navigation"
    else:
        return "Unknown"


def print_section(title: str, char: str = "═", length: int = 60):
    """Pretty print section header"""
    print(f"\n{char * length}")
    print(f"{title.center(length)}")
    print(f"{char * length}\n")
