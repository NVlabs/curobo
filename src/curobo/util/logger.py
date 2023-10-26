#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# Standard Library
import logging


def setup_curobo_logger(level="info"):
    FORMAT = "[%(levelname)s] [%(name)s] %(message)s"
    if level == "info":
        level = logging.INFO
    elif level == "debug":
        level = logging.DEBUG
    elif level == "error":
        level = logging.ERROR
    elif level in ["warn", "warning"]:
        level = logging.WARN
    else:
        raise ValueError("Log level should be one of [info,debug, warn, error]")
    logging.basicConfig(format=FORMAT, level=level)
    logger = logging.getLogger("curobo")
    logger.setLevel(level=level)


def log_warn(txt: str, *args, **kwargs):
    logger = logging.getLogger("curobo")
    logger.warning(txt, *args, **kwargs)


def log_info(txt: str, *args, **kwargs):
    logger = logging.getLogger("curobo")
    logger.info(txt, *args, **kwargs)


def log_error(txt: str, exc_info=True, stack_info=True, *args, **kwargs):
    logger = logging.getLogger("curobo")
    logger.error(txt, exc_info=exc_info, stack_info=stack_info, *args, **kwargs)
    raise
