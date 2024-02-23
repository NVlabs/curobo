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

Results in the arxiv paper were obtained from v0.6.0. 

v0.6.2+ has significant changes to improve motion quality with lower motion time, lower path length, higher pose accuracy (<1mm). v0.6.2+ sacrifices 15ms (RTX 4090) of compute time to achieve signficantly better solutions. The new results are yet to be added to the technical report. For now, to get the latest benchmark results follow instructions here: https://curobo.org/source/getting_started/4_benchmarks.html.

To get results similar to in the technical report, pass `--report_edition` to `curobo_benchmark.py`. 


# Latest Results (Feb 2024)

Motion Generation on 2600 problems from motion benchmaker and motion policy networks, gives the 
following values on a RTX 4090:

| Metric            | Value                                                        |
|-------------------|--------------------------------------------------------------|
|Success %          | 99.84                                                        |
|Plan Time (s)      | mean: 0.068 ± 0.158 median:0.042 75%: 0.055 98%: 0.246       |
|Motion Time (s)    | mean: 1.169 ± 0.360 median:1.140 75%: 1.381 98%: 2.163       |
|Path Length (rad.) | mean: 3.177 ± 1.072 median:3.261 75%: 3.804 98%: 5.376       |
|Jerk               | mean: 97.700 ± 48.630 median:88.993 75%: 126.092 98%: 199.540|
|Position Error (mm)| mean: 0.119 ± 0.341 median:0.027 75%: 0.091 98%: 1.039       |


## Motion Benchmaker (800 problems):

| Metric            | Value                                                        |
|-------------------|--------------------------------------------------------------|
|Success %          | 100                                                          |
|Plan Time (s)      | mean: 0.063 ± 0.137 median:0.042 75%: 0.044 98%: 0.206       |
|Motion Time (s)    | mean: 1.429 ± 0.330 median:1.392 75%: 1.501 98%: 2.473       |
|Path Length (rad.) | mean: 3.956 ± 0.783 median:3.755 75%: 4.352 98%: 6.041       |
|Jerk               | mean: 67.284 ± 27.788 median:61.853 75%: 83.337 98%: 143.118 |
|Position Error (mm)| mean: 0.079 ± 0.139 median:0.032 75%: 0.077 98%: 0.472       |


## Motion Policy Networks (1800 problems):

| Metric            | Value                                                           |
|-------------------|-----------------------------------------------------------------|
|Success %          | 99.77                                                           |
|Plan Time (s)      | mean: 0.068 ± 0.117 median:0.042 75%: 0.059 98%: 0.243          |
|Motion Time (s)    | mean: 1.051 ± 0.306 median:1.016 75%: 1.226 98%: 1.760          |
|Path Length (rad.) | mean: 2.829 ± 1.000 median:2.837 75%: 3.482 98%: 4.905          |
|Jerk               | mean: 110.610 ± 47.586 median:105.271 75%: 141.517 98%: 217.158 |
|Position Error (mm)| mean: 0.122 ± 0.343 median:0.024 75%: 0.095 98%: 1.114          |
