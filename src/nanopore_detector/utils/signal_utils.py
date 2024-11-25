# utils/signal_utils.py

import numpy as np
from typing import Tuple
from scipy.signal import savgol_filter
import logging
from dataclasses import dataclass

from tqdm import tqdm

logger = logging.getLogger('signal_utils')

# def remove_anomalies(
#     x: np.ndarray,
#     y: np.ndarray,
#     baseline_window: int = 20000,
#     major_std_threshold: float = 7.0,
#     minor_std_threshold: float = 1.3,
#     context_points: int = 200,
#     smoothing_window: int = 21,
#     polyorder: int = 2,
#     plot_steps: bool = False
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """Remove anomalies from signal data."""
#     try:
#         # Copy arrays to avoid modifying originals
#         x_clean = x.copy()
#         y_clean = y.copy()
        
#         # Calculate baseline statistics
#         baseline_mean = np.mean(y_clean[:baseline_window])
#         baseline_std = np.std(y_clean[:baseline_window])
        
#         # Remove major anomalies
#         major_mask = np.abs(y_clean - baseline_mean) > (major_std_threshold * baseline_std)
#         major_indices = np.where(major_mask)[0]
        
#         for idx in major_indices:
#             start_idx = max(0, idx - context_points)
#             end_idx = min(len(y_clean), idx + context_points)
#             local_mean = np.mean(y_clean[start_idx:end_idx][~major_mask[start_idx:end_idx]])
#             y_clean[idx] = local_mean
        
#         # Apply smoothing
#         y_smooth = savgol_filter(y_clean, smoothing_window, polyorder)
        
#         # Remove minor anomalies
#         baseline_std_smooth = np.std(y_smooth[:baseline_window])
#         minor_mask = np.abs(y_clean - y_smooth) > (minor_std_threshold * baseline_std_smooth)
#         y_clean[minor_mask] = y_smooth[minor_mask]
        
#         return x_clean, y_clean
        
#     except Exception as e:
#         logger.error(f"Error in remove_anomalies: {str(e)}")
#         raise

def remove_anomalies(x: np.ndarray, y: np.ndarray, 
                    baseline_window: int = 10000,
                    major_std_threshold: float = 10.0,
                    minor_std_threshold: float = 0.3,
                    context_points: int = 200,
                    smoothing_window: int = 21,
                    polyorder: int = 3,
                    plot_steps: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clean nanopore data by removing clogging events and upward deviations.
    
    Parameters:
    -----------
    x : np.ndarray
        Time data
    y : np.ndarray
        Current measurements
    baseline_window : int
        Number of points to use for baseline calculation
    major_std_threshold : float
        Number of standard deviations for identifying major clogging events
    minor_std_threshold : float
        Number of standard deviations for identifying upward deviations
    context_points : int
        Number of points before and after each event to also replace
    """
    if len(x) != len(y):
        raise ValueError("x and y arrays must have the same length")
    if baseline_window > len(y):
        baseline_window = len(y)
        
    # Calculate baseline statistics
    baseline_median = np.median(y[:baseline_window])
    baseline_std = np.std(y[:baseline_window])
    
    # Create copy of data for cleaning
    y_cleaned = np.copy(y)
    
    # Identify major deviations (clogging events)
    major_deviations = (y > baseline_median + major_std_threshold * baseline_std) | \
                      (y < baseline_median - major_std_threshold * baseline_std)
    
    # Identify minor upward deviations
    minor_upward = (y > baseline_median + minor_std_threshold * baseline_std)
    
    # Combine masks
    all_deviations = major_deviations | minor_upward
    
    # Add context points
    expanded_mask = np.copy(all_deviations)
    for i in range(1, context_points + 1):
        expanded_mask[i:] |= all_deviations[:-i]  # Add points after
        expanded_mask[:-i] |= all_deviations[i:]  # Add points before
    
    # Replace deviations with baseline
    y_cleaned[expanded_mask] = baseline_median * 0.75
    return x, y_cleaned

@dataclass
class ClogStats:
    """Statistics for detected anomalies"""
    clog_starts: np.ndarray
    clog_ends: np.ndarray
    clog_durations: np.ndarray
    total_points_replaced: int
    baseline: float

def replace_clogs_chunked(x: np.ndarray, y: np.ndarray,
                         chunk_size: int = 1_000_000,
                         clog_threshold: float = 0.7,
                         voltage_jump_threshold: float = 1.2,
                         min_clog_duration: int = 100,
                         context_points: int = 200,
                         overlap: int = 1000) -> Tuple[np.ndarray, np.ndarray, ClogStats]:
    """
    Replace clogging events and voltage-induced jumps with baseline in nanopore data.
    
    Parameters:
    -----------
    x : np.ndarray
        Time data
    y : np.ndarray
        Current measurements
    chunk_size : int
        Size of chunks to process at once
    clog_threshold : float
        Fraction of baseline to identify clogs (e.g., 0.7 means current < 70% of baseline)
    voltage_jump_threshold : float
        Fraction of baseline to identify voltage jumps (e.g., 1.2 means current > 120% of baseline)
    min_clog_duration : int
        Minimum number of points to consider as anomaly
    context_points : int
        Points before/after anomaly to replace
    overlap : int
        Overlap between chunks
    """
    if len(x) != len(y):
        raise ValueError("x and y arrays must have the same length")
    
    # Calculate initial baseline from first chunk that's not clogged
    baseline = None
    
    for i in range(0, min(len(y), chunk_size * 3), chunk_size):
        chunk = y[i:i+chunk_size]
        temp_baseline = np.median(chunk)
        temp_std = np.std(chunk)
        good_points = (chunk > (temp_baseline - 2*temp_std)) & (chunk < (temp_baseline + 2*temp_std))
        if np.sum(good_points) > len(chunk) * 0.8:
            baseline = np.median(chunk[good_points])
            break
    
    if baseline is None:
        raise ValueError("Could not find stable baseline in first few chunks")

    # Create copy of data for cleaning
    y_cleaned = np.copy(y)
    
    # Initialize lists for statistics
    all_clog_starts = []
    all_clog_ends = []
    all_clog_durations = []
    total_points_replaced = 0
    
    # Process data in chunks with overlap
    chunk_starts = list(range(0, len(y) - chunk_size + overlap, chunk_size - overlap))
    if len(y) - chunk_starts[-1] > overlap:
        chunk_starts.append(len(y) - chunk_size)
    
    for chunk_start in tqdm(chunk_starts, desc="Processing chunks"):
        # Get chunk with overlap
        chunk_end = min(chunk_start + chunk_size, len(y))
        y_chunk = y[chunk_start:chunk_end]
        
        # Find both clogs and voltage jumps
        is_clog = y_chunk < (baseline * clog_threshold)
        is_voltage_jump = y_chunk > (baseline * voltage_jump_threshold)
        is_anomaly = is_clog | is_voltage_jump
        
        # Find continuous anomaly regions
        anomaly_changes = np.diff(is_anomaly.astype(int))
        chunk_anomaly_starts = np.where(anomaly_changes == 1)[0] + 1
        chunk_anomaly_ends = np.where(anomaly_changes == -1)[0] + 1
        
        # Handle edge cases
        if is_anomaly[0]:
            chunk_anomaly_starts = np.insert(chunk_anomaly_starts, 0, 0)
        if is_anomaly[-1]:
            chunk_anomaly_ends = np.append(chunk_anomaly_ends, len(y_chunk))
            
        # Calculate durations
        chunk_anomaly_durations = chunk_anomaly_ends - chunk_anomaly_starts
        
        # Filter short fluctuations
        valid_anomalies = chunk_anomaly_durations >= min_clog_duration
        chunk_anomaly_starts = chunk_anomaly_starts[valid_anomalies]
        chunk_anomaly_ends = chunk_anomaly_ends[valid_anomalies]
        chunk_anomaly_durations = chunk_anomaly_durations[valid_anomalies]
        
        # Expand anomaly regions
        expanded_starts = np.maximum(chunk_anomaly_starts - context_points, 0)
        expanded_ends = np.minimum(chunk_anomaly_ends + context_points, len(y_chunk))
        
        # Replace anomalies with baseline
        for start, end in zip(expanded_starts, expanded_ends):
            y_cleaned[chunk_start + start:chunk_start + end] = baseline
            total_points_replaced += end - start
        
        # Store anomaly statistics (avoiding double-counting in overlap regions)
        if chunk_start == 0:  # First chunk
            valid_region = slice(None, -overlap if len(chunk_starts) > 1 else None)
        elif chunk_start == chunk_starts[-1]:  # Last chunk
            valid_region = slice(overlap, None)
        else:  # Middle chunks
            valid_region = slice(overlap, -overlap)
            
        valid_anomalies = (chunk_anomaly_starts >= valid_region.start if valid_region.start else 0) & \
                         (chunk_anomaly_ends <= valid_region.stop if valid_region.stop else len(y_chunk))
        
        all_clog_starts.extend(chunk_anomaly_starts[valid_anomalies] + chunk_start)
        all_clog_ends.extend(chunk_anomaly_ends[valid_anomalies] + chunk_start)
        all_clog_durations.extend(chunk_anomaly_durations[valid_anomalies])
    
    stats = ClogStats(
        np.array(all_clog_starts),
        np.array(all_clog_ends),
        np.array(all_clog_durations),
        total_points_replaced,
        baseline
    )
    
    return x, y_cleaned, stats


def calculate_baseline(y: np.ndarray, baseline_window: int = 2000) -> Tuple[float, float]:
    """Calculate baseline and noise standard deviation."""
    try:
        # Calculate baseline using initial segment
        baseline = np.percentile(y[:baseline_window], 10)
        
        # Calculate noise std using values below median
        median = np.median(y[:baseline_window])
        noise_values = y[:baseline_window][y[:baseline_window] < median]
        noise_std = np.std(noise_values)
        
        return baseline, noise_std
        
    except Exception as e:
        logger.error(f"Error in calculate_baseline: {str(e)}")
        raise

def get_adaptive_threshold(amplitude: float) -> float:
    """Get adaptive threshold based on amplitude."""
    try:
        if amplitude > 100:
            return 0.002
        elif amplitude > 50:
            return 0.001
        else:
            return 0.0005
    except Exception as e:
        logger.error(f"Error in get_adaptive_threshold: {str(e)}")
        raise

def invert_signal(y: np.ndarray) -> np.ndarray:
    """Invert signal for negative-going events."""
    return y * (-1)