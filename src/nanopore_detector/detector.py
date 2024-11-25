# detector.py

from typing import Optional
import numpy as np
import pandas as pd
import os
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time



from nanopore_detector.config.config import Config
from nanopore_detector.processors.signal_processor import SignalProcessor
from nanopore_detector.processors.event_processor import EventProcessor
from nanopore_detector.processors.filter_processor import FilterProcessor

global abf

from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd
import os
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
from functools import lru_cache
from collections import deque

class EventDetector:
    """Main event detector that coordinates signal processing and event detection"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger('EventDetector')
        
        # Initialize processors with caching
        self.signal_processor = SignalProcessor(config)
        self.event_processor = EventProcessor(config)
        self.filter_processor = FilterProcessor(config)
        
        # Pre-allocate reusable arrays
        self._initialize_buffers()
    
    def _initialize_buffers(self):
        """Pre-allocate buffers for commonly used arrays"""
        self.window_buffer = np.zeros(int(self.config.window_size * 1.2))  # 20% buffer for safety
        self.result_buffer = deque(maxlen=10000)  # Limit memory usage for results
    
    @lru_cache(maxsize=128)
    def _calculate_window_parameters(self, freq: int, sampling_rate: float) -> Tuple[float, float, float, int]:
        """Cache frequently calculated window parameters"""
        all_time = freq / sampling_rate
        point_time = all_time / freq
        split_time = self.config.window_size / 1000
        split_points = int(freq * split_time / all_time)
        return all_time, point_time, split_time, split_points

    def detect_events(
        self,
        file_path: str,
        config: dict,
        output_path: Optional[str] = None,
        debug: bool = False,
        
    ) -> pd.DataFrame:
        """
        Optimized event detection function with enhanced multithreading.
        """
        if debug:
            self.logger.info("Starting event detection...")

        try:
            # Load and prepare data using vectorized operations
            x, y, sampling_rate, global_baseline, global_noise_std = \
                self.signal_processor.load_and_prepare(debug=False)

            # Use cached window parameters
            all_time, point_time, split_time, split_points = self._calculate_window_parameters(
                len(x), sampling_rate
            )

            if debug:
                self.logger.info(f'Total points: {len(x)}')
                self.logger.info(f'Total duration: {all_time:.2f}s')
                self.logger.info(f'Number of slices: {int(all_time / split_time)}')

            # Generate window positions more efficiently
            positions = np.arange(0, len(x), split_points)
            positions = np.append(positions, len(x))
            # self.logger.info(f'Total windows: {len(positions)}')
            window_limit = config.window_limit
            if config.window_limit > 0:  # Only if window_to_see is specified
                positions = positions[:window_limit + 1]  # +1 because we need end position
                # self.logger.info(f"Processing first {window_limit} windows only") 
            

            # Optimize batch size based on available cores
            optimal_batch_size = max(1, len(positions) // (self.config.max_workers * 4))
            
            # Process windows in parallel with optimized batching
            results = {}
            all_amplitudes = []
            all_peaks = []

            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = []
                
                # Create optimized batches
                for i in range(0, len(positions) - 1, optimal_batch_size):
                    end_idx = min(i + optimal_batch_size, len(positions) - 1)
                    batch_indices = [
                        (positions[j], positions[j + 1]) 
                        for j in range(i, end_idx)
                    ]
                    
                    future = executor.submit(
                        self.event_processor.process_window_batch,
                        batch_indices,
                        x,
                        y,
                        point_time,
                        global_baseline,
                        global_noise_std,
                        debug
                    )
                    futures.append((future, i))

                # Process results as they complete
                for future, batch_idx in tqdm(futures, desc="Processing windows"):
                    try:
                        batch_results, batch_amplitudes, batch_peaks = future.result()
                        results.update(batch_results)
                        all_amplitudes.extend(batch_amplitudes)
                        all_peaks.extend(batch_peaks)
                    except Exception as e:
                        self.logger.warning(f"Error processing batch {batch_idx}: {str(e)}")
                        continue

            if debug:
                self.logger.info(f'Raw events detected: {len(results)}')

            if not results:
                if debug:
                    self.logger.info("No events detected!")
                return pd.DataFrame()

            # Process results with vectorized operations
            file_index = os.path.splitext(os.path.basename(file_path))[0]
            events_df = self.filter_processor.process_results(
                results=results,
                all_amplitudes=all_amplitudes,
                file_index=file_index,
                debug=False
            )

            # Vectorized duration filtering
            if hasattr(self.config, 'duration_filter') and self.config.duration_filter:
                events_df = self.filter_processor.add_duration_filtering(events_df)

            return events_df

        except Exception as e:
            if debug:
                self.logger.error(f"Error in event detection: {str(e)}")
            return pd.DataFrame()

    def process_file(
        self,
        file_path: str,
        output_path: Optional[str] = None,
        window_to_see: int = -1,
        debug: bool = False
    ) -> pd.DataFrame:
        """
        Process a single file with optimized timing and logging.
        """
        start_time = time.time()
        
        try:
            events_df = self.detect_events(
                file_path=file_path,
                config=self.config,
                output_path=output_path,
                debug=debug
            )
            
            end_time = time.time()
            processing_time = round(end_time - start_time, 3)
            
            if not events_df.empty:
                # Use vectorized operations for statistics
                stats = {
                    'count': len(events_df),
                    'mean_amplitude': events_df['Amplitude (nA)'].mean(),
                    'mean_duration': events_df['Duration (ms)'].mean()
                }
                
                self.logger.info(f'\nProcessing completed in {processing_time}s')
                self.logger.info(f"\nDetected {stats['count']} events")
                self.logger.info("\nEvent statistics:")
                self.logger.info(f"Mean amplitude: {stats['mean_amplitude']:.2f} nA")
                self.logger.info(f"Mean duration: {stats['mean_duration']:.2f} ms")
            else:
                self.logger.info(f'\nProcessing completed in {processing_time}s')
                self.logger.info("No events detected")
            
            return events_df
            
        except Exception as e:
            self.logger.error(f"Error processing file: {str(e)}")
            raise