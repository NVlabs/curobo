<!--
Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
property and proprietary rights in and to this material, related
documentation and any modifications thereto. Any use, reproduction,
disclosure or distribution of this material and related documentation
without an express license agreement from NVIDIA CORPORATION or
its affiliates is strictly prohibited.
-->
This folder contains scripts to run the motion planning benchmarks.

Refer to Benchmarks & Profiling instructions: https://curobo.org/source/getting_started/4_benchmarks.html.

### Note

Results in the arxiv paper were obtained from v0.6.0. 

v0.6.2+ has significant changes to improve motion quality with lower motion time, lower path length, higher pose accuracy (<1mm). v0.6.2+ sacrifices 30ms (RTX 4090) of compute time to achieve signficantly better solutions. The new results are yet to be added to the technical report. For now, to get the latest benchmark results follow instructions here: https://curobo.org/source/getting_started/4_benchmarks.html.

To get results similar to in the technical report, pass `--report_edition` to `curobo_benchmark.py`. 