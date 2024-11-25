import pytest
from nanopore_detector import EventDetector
from nanopore_detector.config import Config

def test_detector_initialization():
    config = Config()
    detector = EventDetector(config)
    assert detector is not None

def test_process_file():
    config = Config()
    detector = EventDetector(config)
    # Add your test cases here
