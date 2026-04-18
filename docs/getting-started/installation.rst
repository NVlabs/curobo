.. _tutorial_installation:

Installation
============

This tutorial walks you through installing cuRobo on Ubuntu 22.04 with PyTorch.
Follow these steps exactly for a guaranteed working installation.

System Requirements
-------------------

Before starting, ensure you have:

1. **Ubuntu>=20.04** (other Linux environments may work)
2. **NVIDIA GPU > Turing** and at least 4 GB VRAM
3. **NVIDIA Driver >= 580.65.06** (driver should support at least CUDA 12)
4. **Python>=3.10** (> 3.13 is not validated)

Installation
------------------

We recommend using `uv <https://docs.astral.sh/uv/getting-started/installation/>`_ for
installation.

Step 1: Clone cuRobo
~~~~~~~~~~~~~~~~~~~~~

.. parsed-literal::

   git clone |repo_url| && cd curobo

Step 2: Create a virtual environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   uv venv --python 3.11
   source .venv/bin/activate

Step 3: Install cuRobo
~~~~~~~~~~~~~~~~~~~~~~~~

Check your driver's CUDA version with ``nvidia-smi | grep CUDA``, then pick the
matching install command:

**CUDA 13.x:**

.. code-block:: bash

   uv pip install .[cu13-torch]       # fresh install (includes PyTorch)
   uv pip install .[cu13]             # if PyTorch is already installed

**CUDA 12.x:**

.. code-block:: bash

   uv pip install .[cu12-torch]       # fresh install (includes PyTorch)
   uv pip install .[cu12]             # if PyTorch is already installed

Step 4: Test your installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -c "import curobo; print(curobo.__version__)"
   pytest --pyargs curobo.tests

Third-Party Software
--------------------

This project will download and install additional third-party open source software projects.
Review the license terms of these open source projects before use.
