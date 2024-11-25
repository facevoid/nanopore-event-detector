Quick Start Guide
===============

Basic Usage
----------

This guide will help you get started with the Nanopore Event Detector package.

Installation
-----------

Install the package using pip:

.. code-block:: console

    pip install nanopore-event-detector

Example Usage
-----------

Here's a simple example to get you started:

.. code-block:: python

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

    print(f"Detected {len(events_df)} events")