# processors/signal_processor.py

import numpy as np
import pyabf
import logging
from pathlib import Path
from typing import Tuple
import os
import traceback
import sys
# sys.path.append('/raid/scratch/mxs2361/projects/nanopore-event-visualization/event_detection')
import os

# path_to_check = '/raid/scratch/mxs2361/projects/nanopore-event-visualization/event_detection'

# if os.path.exists(path_to_check):
#     modules = []
#     for entry in os.listdir(path_to_check):
#         full_path = os.path.join(path_to_check, entry)
#         # Check for Python files or directories that are packages
#         if entry.endswith('.py') and not entry.startswith('__'):
#             modules.append(entry[:-3])  # Remove the `.py` extension
#         elif os.path.isdir(full_path) and '__init__.py' in os.listdir(full_path):
#             modules.append(entry)  # Add package names
#     print("Available modules and packages:")
#     print(modules)
# else:
#     print(f"The path '{path_to_check}' does not exist.")
from nanopore_detector.utils.signal_utils import remove_anomalies, calculate_baseline, invert_signal, replace_clogs_chunked
from nanopore_detector.utils.file_utils import save_cleaned_signal
from nanopore_detector.config.config import Config
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

global abf

class SignalProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger('SignalProcessor')

    def apply_filters(self, y: np.ndarray, debug: bool = False) -> np.ndarray:
        """Apply Savitzky-Golay and Gaussian filters to smooth the signal."""
        try:
            if debug:
                self.logger.info("Applying filters to signal...")
                self.logger.info(f"Initial signal range: [{np.min(y):.2f}, {np.max(y):.2f}]")
            
            # Apply Savitzky-Golay filter to reduce noise
            sg_filtered = savgol_filter(y, 
                                    window_length=self.config.savgol_window,
                                    polyorder=self.config.savgol_polyorder)
            
            # Apply gentle Gaussian smoothing
            smoothed_signal = gaussian_filter1d(sg_filtered, sigma=self.config.gaussian_sigma)
            
            if debug:
                self.logger.info(f"Final smoothed range: [{np.min(smoothed_signal):.2f}, {np.max(smoothed_signal):.2f}]")
            
            return smoothed_signal
            
        except Exception as e:
            self.logger.error(f"Error in apply_filters: {str(e)}")
            raise
    
    # def calculate_smoothing_params(self, sampling_rate: int = 250000, segment_time: float = 0.5) -> dict:
    #     """
    #     Calculate smoothing parameters following EVENTPRO logic.
        
    #     Args:
    #         sampling_rate: Sampling frequency in Hz
    #         segment_time: Time window in seconds
        
    #     Returns:
    #         Dictionary containing calculated parameters
    #     """
    #     points_in_window = sampling_rate * segment_time
    #     step_size = int((points_in_window/125000) * 200)
    #     gauss_window = step_size * 100
    #     sigma = gauss_window / 5
        
    #     return {
    #         'points_in_window': points_in_window,
    #         'step_size': step_size,
    #         'gauss_window': gauss_window,
    #         'sigma': sigma
    #     }
    
    # def apply_filters(self, y: np.ndarray, sampling_rate: int = 250000, segment_time: float = 0.5, debug: bool = False) -> np.ndarray:
    #     """
    #     Apply Gaussian smoothing matching the MATLAB EVENTPRO approach.
        
    #     Args:
    #         y: Input signal
    #         sampling_rate: Sampling frequency in Hz (default 250kHz)
    #         segment_time: Time window in seconds (default 0.5s)
    #         debug: Enable debug logging
    #     """
    #     try:
    #         if debug:
    #             self.logger.info("Applying EVENTPRO-style Gaussian smoothing...")
    #             self.logger.info(f"Initial signal range: [{np.min(y):.2f}, {np.max(y):.2f}]")
            
    #         # Calculate window size following EVENTPRO logic
    #         points_in_window = sampling_rate * segment_time
    #         step_size = int((points_in_window/125000) * 200)
    #         gauss_window = step_size * 100
            
    #         # Calculate sigma (MATLAB smoothdata uses window_size/5 for gaussian)
    #         sigma = gauss_window / 5
            
    #         if debug:
    #             self.logger.info(f"Calculated parameters:")
    #             self.logger.info(f"Window points: {points_in_window}")
    #             self.logger.info(f"Step size: {step_size}")
    #             self.logger.info(f"Gaussian window: {gauss_window}")
    #             self.logger.info(f"Sigma: {sigma}")
            
    #         # Apply Gaussian smoothing
    #         smoothed_signal = gaussian_filter1d(y, sigma=sigma)
            
    #         if debug:
    #             self.logger.info(f"Final smoothed range: [{np.min(smoothed_signal):.2f}, {np.max(smoothed_signal):.2f}]")
            
    #         return smoothed_signal

    #     except Exception as e:
    #         self.logger.error(f"Error in apply_filters: {str(e)}")
    #         raise
     
    def load_and_prepare(self, debug: bool = False) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
        """Load and prepare signal data."""
        try:
            if debug:
                self.logger.info("Loading data...")
                
            # Load ABF file
            global abf
            abf = pyabf.ABF(str(self.config.file_path))
            
            if debug:
                self.logger.info(f"Data loaded - points: {len(abf.sweepY)}, "
                               f"range: [{np.min(abf.sweepY):.2f}, {np.max(abf.sweepY):.2f}]")
            
            # Clean signal
            

            x, y, stats = replace_clogs_chunked(
                abf.sweepX, abf.sweepY,
                chunk_size=self.config.chunk_size,  # Adjust if needed
                clog_threshold=self.config.clog_threshold,    # Adjust based on your clog depth
                min_clog_duration=self.config.min_clog_duration, # Minimum points for a clog
                voltage_jump_threshold= self.config.voltage_jump_threshold,  # New parameter for upper bound
                context_points=self.config.context_points,    # Points to remove before/after clog
                overlap=self.config.overlap          # Overlap between chunks
            )
            # x, y = remove_anomalies(abf.sweepX, abf.sweepY)
            y = self.apply_filters(y)
            abf.sweepY = y
            
            # Handle signal direction
            if self.config.signal_direction == 1:
                # if debug:
                #     self.logger.info("Inverting signal for negative-going events")
                #     self.logger.info(f"Signal range before inversion: [{np.min(y):.2f}, {np.max(y):.2f}]")
                # y = invert_signal(y)
                if debug:
                    # self.logger.info(f"Signal range after inversion: [{np.min(y):.2f}, {np.max(y):.2f}]")
                    self.logger.info(f"Found {len(stats.clog_durations)} clogs")
                    self.logger.info(f"Total points replaced: {stats.total_points_replaced}")
                    self.logger.info(f"Percentage of data replaced: {stats.total_points_replaced/len(x)*100:.2f}%")
                    self.logger.info(f"Baseline current: {stats.baseline:.2f}")
            
            # Calculate baseline
            global_baseline, global_noise_std = calculate_baseline(y)
            
            if debug:
                self.logger.info("\nBaseline calculation debug:")
                self.logger.info(f"Initial segment length: 20000")
                self.logger.info(f"Initial segment range: [{np.min(y[:20000]):.2f}, {np.max(y[:20000]):.2f}]")
                self.logger.info(f"Calculated baseline: {global_baseline:.2f}")
                self.logger.info(f"Calculated noise_std: {global_noise_std:.4f}")
                self.logger.info(f'Sampling rate: {abf.dataRate} Hz')
            
            # Save cleaned signal if output path provided
            if self.config.output_path:
                save_cleaned_signal(y, self.config.output_path, self.config.file_path)
                
            return x, y, abf.dataRate, global_baseline, global_noise_std
            
        except Exception as e:
            # self.logger.error(f"Error in load_and_prepare: {str(e)}")
            self.logger.error(f"Error in load_and_prepare: {str(e)}\n{traceback.format_exc()}")
            raise

    def get_signal_info(self, y: np.ndarray, sampling_rate: float, debug: bool = False):
        """Calculate and log signal information."""
        try:
            freq = len(y)
            all_time = freq / sampling_rate
            if debug:
                self.logger.info(f'Total points: {freq}')
                self.logger.info(f'Total duration: {all_time:.2f}s')
                self.logger.info(f'Number of slices: {int(all_time / (self.config.window_size / 1000))}')
            return freq, all_time
        except Exception as e:
            self.logger.error(f"Error getting signal info: {str(e)}")
            raise