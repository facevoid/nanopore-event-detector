# Nanopore Event Detector

[![Documentation Status](https://readthedocs.org/projects/nanopore-event-detector/badge/?version=latest)](https://nanopore-event-detector.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/nanopore-event-detector.svg)](https://badge.fury.io/py/nanopore-event-detector)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A robust and efficient Python package for real-time detection of molecular translocation events in nanopore signals.

## Features

- Real-time event detection with adaptive thresholding
- Robust baseline tracking and drift correction
- Automated clog detection and removal
- Parallel processing support
- Comprehensive event validation framework

## Installation

```bash
pip install nanopore-event-detector
```

## Quick Start

```python
from nanopore_detector import EventDetector
from nanopore_detector.config import Config

# Create configuration
config = Config()

# Initialize detector
detector = EventDetector(config)

# Process file
events_df = detector.process_file(
    file_path="your_data.abf",
    output_path="results/",
    debug=True
)
```

## Documentation

Full documentation is available at [Read the Docs](https://nanopore-event-detector.readthedocs.io/).

## Citation

If you use this package in your research, please cite:

```bibtex
@article{your-paper-2024,
    title={Adaptive Real-Time Event Detection for Nanopore Sensing},
    author={Your Name et al.},
    journal={Journal Name},
    year={2024}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.