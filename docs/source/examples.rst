Usage Examples
============

Basic Examples
------------

Signal Processing
^^^^^^^^^^^^^^

.. code-block:: python

    from nanopore_detector.processors import SignalProcessor
    from nanopore_detector.config import Config

    config = Config()
    processor = SignalProcessor(config)
    
    # Process signal
    x, y, sampling_rate, baseline, noise = processor.load_and_prepare(debug=True)

Event Detection
^^^^^^^^^^^^

.. code-block:: python

    from nanopore_detector import EventDetector
    from nanopore_detector.config import Config

    config = Config()
    detector = EventDetector(config)
    
    # Configure parameters
    config.window_size = 35  # milliseconds
    config.baseline_std_multiplier = 4.0
    
    # Detect events
    events_df = detector.process_file("data.abf", "output/")

Advanced Examples
--------------

Custom Configuration
^^^^^^^^^^^^^^^^

.. code-block:: python

    from nanopore_detector.config import Config
    
    config = Config()
    
    # Customize processing parameters
    config.chunk_size = 1000000
    config.clog_threshold = 0.7
    config.savgol_window = 11
    config.gaussian_sigma = 0.5

Parallel Processing
^^^^^^^^^^^^^^^

.. code-block:: python

    config.max_workers = 8  # Set number of parallel workers
    detector = EventDetector(config)