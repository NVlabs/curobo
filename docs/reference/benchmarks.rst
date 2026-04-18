Benchmarks & Profiling
=======================

Latest Motion Generation Results
*********************************
Results obtained on November 22 2024 (v0.7.6).

Motion Generation on 2600 problems from motion benchmaker and motion policy networks, on a
RTX 6000 Ada:

+---------------------+---------------------------------------------------------------------+
| Metric              | Value                                                               |
+=====================+=====================================================================+
| Success %           | 99.73                                                               |
+---------------------+---------------------------------------------------------------------+
| Plan Time (s)       | mean: 0.038 ± 0.014  median: 0.036  75%: 0.041  98%: 0.082          |
+---------------------+---------------------------------------------------------------------+
| Solve Time (s)      | mean: 0.031 ± 0.012  median: 0.029  75%: 0.032  98%: 0.065          |
+---------------------+---------------------------------------------------------------------+
| Position Error (mm) | mean: 0.037 ± 0.301  median: 0.000  75%: 0.000  98%: 0.348          |
+---------------------+---------------------------------------------------------------------+
| Path Length (rad.)  | mean: 3.134 ± 1.056  median: 3.254  75%: 3.828  98%: 5.140          |
+---------------------+---------------------------------------------------------------------+
| Motion Time(s)      | mean: 1.252 ± 0.364  median: 1.243  75%: 1.486  98%: 2.152          |
+---------------------+---------------------------------------------------------------------+
| Jerk                | mean: 227.028 ± 84.048  median: 213.743  75%: 267.657  98%: 465.272 |
+---------------------+---------------------------------------------------------------------+
| Energy (J)          | mean: 89.655 ± 50.437  median: 79.280  75%: 117.308  98%: 208.464   |
+---------------------+---------------------------------------------------------------------+
| Torque (N·m)        | mean: 70.935 ± 24.882  median: 67.328  75%: 86.330  98%: 130.007    |
+---------------------+---------------------------------------------------------------------+

With torque limits at full payload of 3kg:
+---------------------+---------------------------------------------------------------------+
| Metric              | Value                                                               |
+=====================+=====================================================================+
| Success %           | 99.73                                                               |
+---------------------+---------------------------------------------------------------------+
| Plan Time (s)       | mean: 0.052 ± 0.037  median: 0.044  75%: 0.053  98%: 0.133          |
+---------------------+---------------------------------------------------------------------+
| Solve Time (s)      | mean: 0.042 ± 0.030  median: 0.035  75%: 0.044  98%: 0.111          |
+---------------------+---------------------------------------------------------------------+
| Position Error (mm) | mean: 0.042 ± 0.331  median: 0.000  75%: 0.000  98%: 0.329          |
+---------------------+---------------------------------------------------------------------+
| Path Length (rad.)  | mean: 3.278 ± 1.226  median: 3.329  75%: 3.932  98%: 6.522          |
+---------------------+---------------------------------------------------------------------+
| Motion Time(s)      | mean: 1.362 ± 0.554  median: 1.305  75%: 1.543  98%: 3.145          |
+---------------------+---------------------------------------------------------------------+
| Jerk                | mean: 216.411 ± 84.125  median: 203.553  75%: 253.055  98%: 444.250 |
+---------------------+---------------------------------------------------------------------+
| Energy (J)          | mean: 81.744 ± 42.968  median: 72.801  75%: 105.537  98%: 181.633   |
+---------------------+---------------------------------------------------------------------+
| Torque (N·m)        | mean: 62.246 ± 13.617  median: 65.267  75%: 73.294  98%: 82.232     |
+---------------------+---------------------------------------------------------------------+



Latest Inverse Kinematics Results
*********************************
Results obtained on April 17 2026.

Reported errors are 90th percentile. You can run this with ``python benchmark/ik_benchmark.py``.

+----+----------------+---------------+--------------+--------------+----------------------+--------------------------+----------------------+---------------------+--------------------------------+------------------------------------+
|    | robot          |   IK-time(ms) |   Batch-Size |   Success-IK |   Position-Error(mm) |   Orientation-Error(deg) |   C-Free-IK-time(ms) |   Success-C-Free-IK |   Position-Error-C-Free-IK(mm) |   Orientation-Error-C-Free-IK(deg) |
+====+================+===============+==============+==============+======================+==========================+======================+=====================+================================+====================================+
|  0 | unitree_g1.yml |      31.3916  |          100 |          100 |          0.000982541 |              5.60822e-05 |            526.871   |                98.4 |                    0.00795739  |                        0.000371568 |
+----+----------------+---------------+--------------+--------------+----------------------+--------------------------+----------------------+---------------------+--------------------------------+------------------------------------+
|  1 | dual_ur10e.yml |       6.05801 |          100 |          100 |          0.000191174 |              1.12155e-05 |             15.6443  |                99.2 |                    0.000170931 |                        1.03467e-05 |
+----+----------------+---------------+--------------+--------------+----------------------+--------------------------+----------------------+---------------------+--------------------------------+------------------------------------+
|  2 | franka.yml     |       2.60138 |          100 |          100 |          0.000127351 |              8.30968e-06 |              2.72574 |               100   |                    0.000130251 |                        8.50011e-06 |
+----+----------------+---------------+--------------+--------------+----------------------+--------------------------+----------------------+---------------------+--------------------------------+------------------------------------+

Running Benchmarks
********************

We use robometrics to run some of the benchmarks and also provide csv export with pandas.
Install the ``benchmark`` extra to pull in all required dependencies
(``tabulate``, ``pandas``, ``robometrics``, ``pin``, ``seaborn``):

.. code-block:: bash

    pip install -e ".[benchmark]"
    # or, with uv:
    uv sync --extra benchmark

USD output (``--save_usd`` for ``motion_plan_benchmark.py``) additionally requires
``usd-core``. Install both extras together when you need to write ``.usd`` files:

.. code-block:: bash

    pip install -e ".[benchmark,usd]"
    # or, with uv:
    uv sync --extra benchmark --extra usd

.. note::

    Do not install ``usd-core`` if you are running cuRobo inside Isaac Sim;
    Isaac Sim provides its own USD runtime.


Kinematics & Collision Checking
********************************
To measure compute time and peak memory for forward kinematics, pose-cost
gradients, self-collision, and world-collision (with and without CUDA graphs)
across all supported robots, run:

``python benchmark/cost_gradient_benchmark.py``

Results are written to ``benchmark/log/`` as four YAML files (one per stage),
with times reported in milliseconds and memory in MiB.

Inverse Kinematics
********************************

To measure success metrics and compute time for inverse kinematics and collision-free inverse kinematics, run

``python benchmark/ik_benchmark.py --save_path=. --file_name=ik``

This will save the results to ``ik.yml``.

Motion Planning
***************************************

To run benchmarks for motion planning:

``python benchmark/motion_plan_benchmark.py``, optionally with ``--use-dynamics`` to
enable torque-limited motion planning.





