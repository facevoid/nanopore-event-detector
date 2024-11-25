Contributing Guide
===============

Thank you for considering contributing to Nanopore Event Detector!

Development Setup
--------------

1. Clone the repository:

.. code-block:: console

    git clone https://github.com/facevoid/nanopore-event-detector.git
    cd nanopore-event-detector

2. Install development dependencies:

.. code-block:: console

    pip install -r requirements.txt

3. Install the package in editable mode:

.. code-block:: console

    pip install -e .

Running Tests
-----------

Run the test suite using pytest:

.. code-block:: console

    pytest tests/

Code Style
---------

We follow PEP 8 guidelines. Please ensure your code is formatted using black:

.. code-block:: console

    black src/nanopore_detector