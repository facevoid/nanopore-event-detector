import numpy as np
from typing import Dict, List, Optional, Tuple, NamedTuple
import logging
from concurrent.futures import ThreadPoolExecutor

class EventData(NamedTuple):
    """Data structure for event information"""
    time: float
    peak_time: float
    amplitude: float
    baseline: float
    start_time: float
    end_time: float
    duration: float
    rise: float
    decay: float
    start_idx: int
    end_idx: int
    peak_idx: int

import numpy as np

import numpy as np

class BaselineState:
    """Manages baseline tracking and updates with strict drift protection"""
    def __init__(self, initial_baseline: float, max_window_without_update: float, max_history: int, 
                 window_size: float = 0.5):  # Removed update_rate parameter
        self.initial_baseline = initial_baseline
        self.current = initial_baseline
        self.history = []
        self.max_history = max_history
        self.last_update_time = 0
        self.max_time_without_update = window_size * max_window_without_update
        self.drift_threshold = 0.1  # 10% allowance from initial baseline
    
    def update(self, value: float, update_rate: float, current_time: float = None) -> None:
        """Update baseline with new value using exponential moving average"""
        # Skip update if value deviates too much from initial baseline
        if abs(value - self.initial_baseline) > (self.initial_baseline * 0.05):  # 10% threshold
            return
            
        # Calculate proposed new baseline
        proposed_baseline = (1 - update_rate) * self.current + update_rate * value
        
        # Check if the proposed baseline itself would deviate too much
        if abs(proposed_baseline - self.initial_baseline) > (self.initial_baseline * 0.05):
            return
                
        # Only update if we're still within bounds
        self.current = proposed_baseline
        self.history.append(self.current)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        if current_time is not None:
            self.last_update_time = current_time

    def needs_force_update(self, current_time: float) -> bool:
        """Check if baseline needs forced update due to time-based drift"""
        return (current_time - self.last_update_time) > self.max_time_without_update
    
    @property 
    def true_value(self) -> float:
        """Get true baseline value"""
        return self.current

class EventProcessor:
    def __init__(self, config):
        """Initialize event processor with configuration"""
        self.config = config
        self.logger = logging.getLogger('EventProcessor')

    def process_window_batch(
        self, 
        batch_indices: List[Tuple[int, int]], 
        x: np.ndarray, 
        y: np.ndarray, 
        point_time: float,
        global_baseline: float, 
        global_noise_std: float,
        debug: bool = False
    ) -> Tuple[Dict, List[float], List[float]]:
        """Process multiple windows in parallel - Main entry point"""
        all_results = {}
        all_amplitudes = []
        all_peaks = []

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [
                executor.submit(
                    self.process_single_window,
                    start_idx, end_idx, x, y,
                    global_baseline, global_noise_std
                )
                for start_idx, end_idx in batch_indices
            ]

            for future in futures:
                try:
                    results, amplitudes, peaks, _ = future.result()
                    all_results.update(results)
                    all_amplitudes.extend(amplitudes)
                    all_peaks.extend(peaks)
                except Exception as e:
                    if debug:
                        self.logger.warning(f"Error processing window: {str(e)}")

        return all_results, all_amplitudes, all_peaks

    def process_single_window(
        self,
        start_idx: int,
        end_idx: int,
        x: np.ndarray,
        y: np.ndarray,
        global_baseline: float,
        global_noise_std: float
    ) -> Tuple[Dict, List[float], List[float]]:
        """Process a single window of signal data"""
        results = {}
        possible_amplitude = []
        possible_peak = []
        temp_events = []

        try:
            signal = y[start_idx:end_idx]
            if len(signal) < 10:
                return {}, [], []

            # baseline = BaselineState(global_baseline, self.config.max_window_without_update, self.config.max_baseline_history)
            
            
            baseline = BaselineState(
                global_baseline, 
                self.config.max_window_without_update, 
                self.config.max_baseline_history
            )  # Remove the update_rate parameter
            base_threshold = global_noise_std * self.config.baseline_std_multiplier
            
            events = self._detect_events(signal, x, start_idx, baseline, base_threshold)
            if events:
                temp_events.extend(events)

            # Process and merge events
            if temp_events:
                temp_events.sort(key=lambda x: x['time'])
                merged_events = self._merge_nearby_events(temp_events)
                
                # Add merged events to results
                for event in merged_events:
                    self._add_event_to_results(event, results, possible_amplitude, possible_peak)

        except Exception as e:
            self.logger.warning(f"Error in window processing: {str(e)}")

        return results, possible_amplitude, possible_peak, baseline.current

    def _detect_events(
        self,
        signal: np.ndarray,
        x: np.ndarray,
        start_idx: int,
        baseline: BaselineState,
        base_threshold: float
    ) -> List[dict]:
        """Detect events in the signal with adaptive thresholding for different event types"""
        events = []
        in_event = False
        event_start = None
        lowest_point = None
        lowest_value = float('inf')
        event_duration = 0
        i = 0
        
        # Window for local noise calculation
        local_window_size = 50  # Adjust based on your sampling rate

        while i < len(signal):
            current_value = signal[i]
            current_time = x[start_idx + i]

            # Calculate local noise in window
            window_start = max(0, i - local_window_size)
            window_end = min(len(signal), i + local_window_size)
            local_window = signal[window_start:window_end]
            local_noise_std = np.std(local_window)
            
            # Adaptive threshold based on event context
            if in_event:
                if event_duration > self.config.long_event_threshold:
                    # More flexible threshold for compound events
                    current_threshold = base_threshold * 1.2
                else:
                    # Standard threshold for single protein events
                    current_threshold = base_threshold
            else:
                # Use minimum of global and local threshold for detection
                current_threshold = min(base_threshold, 
                                    local_noise_std * self.config.baseline_std_multiplier)

            deviation = baseline.current - current_value

            # Force baseline update if too long without update
            if baseline.needs_force_update(current_time):
                window_start = max(0, i - 20)
                window_values = signal[window_start:i]
                if len(window_values) > 0:
                    stable_value = np.median(window_values)
                    baseline.update(stable_value, self.config.baseline_update_rate, current_time)
                    in_event = False

            # Regular baseline update when not in event
            if not in_event and abs(current_value - baseline.current) < current_threshold:
                baseline.update(current_value, self.config.baseline_update_rate, current_time)

            # Event detection logic
            if not in_event:
                if deviation > current_threshold:
                    event_start = self._find_event_start(
                        signal, i, baseline.current, current_threshold
                    )
                    in_event = True
                    lowest_point = i
                    lowest_value = current_value
                    event_duration = 0

            elif current_value < lowest_value:
                lowest_value = current_value
                lowest_point = i
                event_duration = i - event_start if event_start is not None else 0

            else:
                event_duration = i - event_start if event_start is not None else 0
                event_end = self._check_event_end(
                    signal, i, baseline.true_value, current_threshold, event_duration
                )
                if event_end is not None:
                    event = self._create_event(
                        x, signal, start_idx, event_start, event_end,
                        lowest_point, lowest_value, baseline.true_value
                    )
                    if event is not None:
                        events.append(event)
                    in_event = False
                    i = event_end

            i += 1

        return events
    
    def _find_event_start(
        self,
        signal: np.ndarray,
        current_idx: int,
        baseline: float,
        threshold: float
    ) -> int:
        """Find the start index of an event"""
        look_back = self.config.lookback_points
        event_start = max(0, current_idx - look_back)
        
        for j in range(current_idx-1, max(0, current_idx-look_back), -1):
            if abs(signal[j] - baseline) < threshold :
                event_start = j + 1
                break
                
        return event_start

    def _check_event_end(
        self,
        signal: np.ndarray,
        current_idx: int,
        baseline: float,
        threshold: float,
        event_duration: int
    ) -> Optional[int]:
        """Check if event has ended with adaptive criteria"""
        look_ahead = self.config.lookahead_points
        if current_idx + look_ahead >= len(signal):
            return None

        future_points = signal[current_idx:current_idx + look_ahead]
        
        # For compound events (longer duration), use more relaxed criteria
        if event_duration > self.config.long_event_threshold:
            if np.all(abs(baseline - future_points) < threshold * 1.2):
                return current_idx + look_ahead
        # For single protein events, use strict criteria
        else:
            if np.all(baseline - future_points < threshold):
                return current_idx + look_ahead
                
        return None

    def _create_event(
        self,
        x: np.ndarray,
        signal: np.ndarray,
        start_idx: int,
        event_start: int,
        event_end: int,
        lowest_point: int,
        lowest_value: float,
        baseline: float
    ) -> Optional[dict]:
        """Create event dictionary if amplitude exceeds minimum"""
        amplitude = baseline - lowest_value
        min_amp = self.config.min_amplitude * self.config.min_amplitude_scaling

        if amplitude <= min_amp:
            return None

        peak_index = start_idx + lowest_point
        start_global_idx = start_idx + event_start
        end_global_idx = start_idx + event_end

        event_duration = x[end_global_idx] - x[start_global_idx]
        if event_duration <= 0:
            return None

        return {
            'time': x[peak_index],
            'peak_time': x[peak_index] * 1000,
            'amplitude': amplitude,
            'baseline': baseline,
            'start_time': x[start_global_idx] * 1000,
            'end_time': x[end_global_idx] * 1000,
            'duration': event_duration * 1000,
            'rise': (x[lowest_point] - x[start_global_idx]) * 1000,
            'decay': (x[end_global_idx] - x[lowest_point]) * 1000,
            'start_idx': start_global_idx,
            'end_idx': end_global_idx,
            'peak_idx': peak_index,
        }

    def _merge_nearby_events(self, events: List[dict]) -> List[dict]:
        """Merge events that are close in time"""
        if not events:
            return []

        merged = []
        current_event = events[0]
        
        for next_event in events[1:]:
            if (next_event['time'] - current_event['time'] < self.config.merge_threshold and
                next_event['amplitude'] > current_event['amplitude']):
                current_event = self._merge_events(current_event, next_event)
            else:
                merged.append(current_event)
                current_event = next_event
                
        merged.append(current_event)
        return merged

    def _merge_events(self, current_event: dict, next_event: dict) -> dict:
        """Merge two events, keeping the higher amplitude event's characteristics"""
        return {
            'time': next_event['time'],
            'peak_time': next_event['peak_time'],
            'amplitude': next_event['amplitude'],
            'baseline': next_event['baseline'],
            'start_time': current_event['start_time'],
            'end_time': next_event['end_time'],
            'duration': next_event['end_time'] - current_event['start_time'],
            'rise': next_event['rise'],
            'decay': next_event['decay'],
            'start_idx': current_event['start_idx'],
            'end_idx': next_event['end_idx'],
            'peak_idx': next_event['peak_idx']
        }

    def _add_event_to_results(
        self,
        event: dict,
        results: dict,
        possible_amplitude: list,
        possible_peak: list
    ) -> None:
        """Add event to results and update amplitude/peak lists"""
        results[event['time']] = [
            event['peak_time'],
            event['amplitude'],
            event['baseline'],
            event['start_time'],
            event['end_time'],
            event['duration'],
            event['rise'],
            event['decay'],
            event['start_idx'],
            event['end_idx'],
            event['peak_idx'],
            0.001
        ]
        possible_amplitude.append(event['amplitude'])
        possible_peak.append(event['peak_time'])