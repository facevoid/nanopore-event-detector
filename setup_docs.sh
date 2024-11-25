#!/bin/bash

# Create documentation directory structure
mkdir -p docs/source/{_static,_templates,api}
touch docs/source/api/{detector,processors,config,utils}.rst
touch docs/source/{installation,quickstart,examples,contributing}.rst

# Create Makefile
cat > docs/Makefile << 'EOL'
# Minimal makefile for Sphinx documentation
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
EOL

# Create make.bat for Windows users
cat > docs/make.bat << 'EOL'
@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=source
set BUILDDIR=build

if "%1" == "" goto help

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.http://sphinx-doc.org/
	exit /b 1
)

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

:end
popd
EOL

# Create conf.py
cat > docs/source/conf.py << 'EOL'
# Configuration file for the Sphinx documentation builder
import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

project = 'Nanopore Event Detector'
copyright = '2024, Your Name'
author = 'Your Name'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx_rtd_theme',
    'sphinx.ext.intersphinx',
    'nbsphinx',
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None
EOL

# Create index.rst
cat > docs/source/index.rst << 'EOL'
Welcome to Nanopore Event Detector's documentation!
================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/detector
   api/processors
   api/config
   api/utils
   examples
   contributing

Installation
-----------

To install Nanopore Event Detector, run this command in your terminal:

.. code-block:: console

    pip install nanopore-event-detector

Quick Start
----------

Here's a simple example:

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

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
EOL

chmod +x setup_docs.sh