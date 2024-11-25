# utils/file_utils.py

import os
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger('file_utils')

def save_cleaned_signal(y: np.ndarray, output_path: str, file_name: str):
    """Save cleaned signal to npy file."""
    try:
        os.makedirs(output_path, exist_ok=True)
        file_index = os.path.splitext(os.path.basename(file_name))[0]
        save_path = os.path.join(output_path, f"{file_index}_cleaned.npy")
        np.save(save_path, y)
        # logger.info(f"Saved cleaned signal to {save_path}")
    except Exception as e:
        logger.error(f"Error saving cleaned signal: {str(e)}")
        raise

def save_events(events_df: pd.DataFrame, output_path: str, file_name: str, filtered: bool = True):
    """Save events to CSV file."""
    try:
        os.makedirs(output_path, exist_ok=True)
        file_index = os.path.splitext(os.path.basename(file_name))[0]
        suffix = "_filtered" if filtered else ""
        save_path = os.path.join(output_path, f"{file_index}{suffix}.csv")
        events_df.to_csv(save_path, index=False)
        logger.info(f"Saved events to {save_path}")
    except Exception as e:
        logger.error(f"Error saving events: {str(e)}")
        raise