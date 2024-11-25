# config/config.py
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
@dataclass
class Config:
    # File handling
    base_path =  "/raid/scratch/mxs2361/projects/nanopore-events-classification/Kremen_and_Spike/Kremen + Spike"
    file_path: str = f"{base_path}/66.6nM Spike + 44.5nM Kremen/Experiment 2/100mV/24o10040.abf"
    output_path: Optional[str] = './output_min_amp_180_std_4'

    # Signal processing
    signal_direction: int = 1  # 1 for negative-going events
    window_size: float = 35  # Window size in milliseconds

    # Clogg Removal
    chunk_size: int = 1000000
    clog_threshold: float = 0.7
    voltage_jump_threshold: float = 1.009
    min_clog_duration: int = 100
    context_points: int = 200
    overlap: int = 1000
    window_limit: int = -1
    
    savgol_window: int = 11
    savgol_polyorder: int = 2
    gaussian_sigma: float = 0.5

    # Baseline calculation
    baseline_update_rate: float = 0.5 # For 0.99 * baseline + 0.01 * current
    max_baseline_history: int = 500  # Maximum baseline history points
    max_window_without_update = 5

    #Local noise threshold
    long_event_threshold = 300  # Adjust based on your typical compound event duration
    local_window_size = 50     # Window for local noise calculation
    compound_threshold_relaxation = 1.2  # 20% more relaxed threshold for compounds
    min_amplitude_scaling_compound = 0.8  # 
    local_window_size = 50  
    # Recalibrated for smaller legitimate events
    """
    For biological/scientific signal detection, there are some common statistical approaches for choosing the multiplier:
    Standard statistical significance levels:
    3σ (multiplier = 3): ~99.7% confidence (~0.3% false positive rate)
    4σ (multiplier = 4): ~99.99% confidence (~0.01% false positive rate)
    5σ (multiplier = 5): ~99.99994% confidence (extremely rare false positives)
    """
    baseline_std_multiplier: float = 4  # Use case, base_threshold = global_noise_std * self.config.baseline_std_multiplier, if deviation > base_threshold: then event
    min_amplitude: float = 100.0  # Set below your smallest legitimate events (~300)
    min_amplitude_scaling: float = 1.0  # Keep at 1
    # base_threshold_scaling: float = 0.4  # Keep high for unstable baseline

    # Event processing parameters
    merge_threshold: float = 0.009  # Your 500 microseconds threshold
    # lookback_points: int = 1  # Your look-back window size
    # lookahead_points: int = 1  # Your look-ahead window size
    
    lookahead_points = 25  # 0.1ms (reduced to handle noise better)
    lookback_points = 25

    # Amplitude filtering
    high_amplitude_threshold: float = 2500  # Your threshold for high amplitude detection
    high_amp_factor: float = 0.05  # Your 0.1 factor for high amplitudes
    low_amp_factor: float = 0.01  # Your 0.3 factor for low amplitudes
    max_workers = 8