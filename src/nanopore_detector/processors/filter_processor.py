# processors/filter_processor.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging
import os
from pathlib import Path

from nanopore_detector.config.config import Config
from nanopore_detector.utils.file_utils import save_events

class FilterProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger('FilterProcessor')
        self.columns = [
            'Event index', 'Peak Time (ms)', 'Amplitude (nA)', 'Baseline',
            'Start time (ms)', 'End time (ms)', 'Duration (ms)',
            'Rise (ms)', 'Decay (ms)', 'Event start index', 'Event end index',
            'peak index', 'Threshold used'
        ]

    def process_results(
        self,
        results: Dict,
        all_amplitudes: List[float],
        file_index: str,
        debug: bool = False
    ) -> pd.DataFrame:
        """Process and filter detection results."""
        try:
            if not results:
                if debug:
                    self.logger.warning("No events to process!")
                return pd.DataFrame()

            # Calculate amplitude statistics
            amplitude_Q = np.percentile(all_amplitudes, [10, 50, 90])
            amplitude_q_1, amplitude_median, amplitude_q_3 = amplitude_Q
            amp_IQR = amplitude_q_3 - amplitude_q_1

            if debug:
                self.logger.info(f"Amplitude statistics:")
                self.logger.info(f"  Q1: {amplitude_q_1:.2f}")
                self.logger.info(f"  Median: {amplitude_median:.2f}")
                self.logger.info(f"  Q3: {amplitude_q_3:.2f}")
                self.logger.info(f"  IQR: {amp_IQR:.2f}")

            # Filter results
            filtered_results = []
            true_index = 1

            # Your exact amplitude threshold calculation
            if amplitude_median > self.config.high_amplitude_threshold:
                min_amplitude_threshold = amplitude_median * self.config.high_amp_factor
            else:
                min_amplitude_threshold = amplitude_median * self.config.low_amp_factor

            if debug:
                self.logger.info("\nAmplitude filtering details:")
                self.logger.info(f"Minimum amplitude threshold: {min_amplitude_threshold}")

            # Apply filtering with your exact logic
            for peak_time, values in sorted(results.items()):
                if debug:
                    self.logger.info(f"\nChecking event at {peak_time}:")
                    self.logger.info(f"Values: {values}")
                    self.logger.info(f"Amplitude: {values[1]}")

                if values[1] >= min_amplitude_threshold:
                    filtered_results.append([true_index] + values)
                    true_index += 1
                    if debug:
                        self.logger.info(f"Event accepted: amplitude {values[1]} >= threshold {min_amplitude_threshold}")
                else:
                    if debug:
                        self.logger.info(f"Event rejected: amplitude {values[1]} < threshold {min_amplitude_threshold}")

            if debug:
                self.logger.info(f"Events after amplitude filtering: {len(filtered_results)}")

            # Create DataFrame
            if filtered_results:
                events_df = pd.DataFrame(filtered_results, columns=self.columns)
                
                # Save results if output path provided
                if self.config.output_path:
                    # Save raw results
                    self._save_raw_results(results, file_index)
                    # Save filtered results
                    save_events(events_df, self.config.output_path, file_index, filtered=True)

                return events_df

            return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Error in process_results: {str(e)}")
            raise

    def _save_raw_results(self, results: Dict, file_index: str):
        """Save raw results before filtering."""
        try:
            output_file = os.path.join(self.config.output_path, f"{file_index}.csv")
            
            with open(output_file, 'w', newline='') as f:
                import csv
                f_writer = csv.writer(f)
                f_writer.writerow(self.columns)
                
                true_index = 1
                for peak_time, values in sorted(results.items()):
                    f_writer.writerow([true_index] + values)
                    true_index += 1
                    
            self.logger.info(f"Saved raw results to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving raw results: {str(e)}")
            raise

    def add_duration_filtering(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Apply duration-based filtering if configured."""
        try:
            if events_df.empty:
                return events_df

            # Calculate duration statistics
            dur_Q = np.percentile(events_df['Duration (ms)'], [10, 50, 90])
            dur_q_1, dur_median, dur_q_3 = dur_Q
            dur_IQR = dur_q_3 - dur_q_1

            self.logger.info(f"\nDuration statistics:")
            self.logger.info(f"  Q1: {dur_q_1:.2f}ms")
            self.logger.info(f"  Median: {dur_median:.2f}ms")
            self.logger.info(f"  Q3: {dur_q_3:.2f}ms")
            self.logger.info(f"  IQR: {dur_IQR:.2f}ms")

            # Filter by duration
            filtered_df = events_df[events_df['Duration (ms)'] <= dur_q_3 + 2.0 * dur_IQR].copy()

            # Final statistics
            self.logger.info(f"\nFinal event statistics:")
            self.logger.info(f"  Total events: {len(filtered_df)}")
            self.logger.info(f"  Mean amplitude: {filtered_df['Amplitude (nA)'].mean():.2f}")
            self.logger.info(f"  Mean duration: {filtered_df['Duration (ms)'].mean():.2f}ms")

            return filtered_df

        except Exception as e:
            self.logger.error(f"Error in duration filtering: {str(e)}")
            return events_df