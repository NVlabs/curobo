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
"""
This module provides logging API, wrapping :py:class:`logging.Logger`. These functions are used
to log messages in the cuRobo package. The functions can also be used in other packages by
creating a new logger (:py:meth:`setup_logger`) with the desired name.
"""
# Standard Library
import logging
import sys


def setup_logger(level="info", logger_name: str = "curobo"):
    """Set up logger level.

    Args:
        level: Log level. Default is "info". Other options are "debug", "warning", "error".
        logger_name: Name of the logger. Default is "curobo".

    Raises:
        ValueError: If log level is not one of [info, debug, warning, error].
    """
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
    logger = logging.getLogger(logger_name)
    logger.setLevel(level=level)


def setup_curobo_logger(level="info"):
    """Set up logger level for curobo package. Deprecated. Use :py:meth:`setup_logger` instead."""
    return setup_logger(level, "curobo")


def log_warn(txt: str, logger_name: str = "curobo", *args, **kwargs):
    """Log warning message. Also see :py:meth:`logging.Logger.warning`.

    Args:
        txt: Warning message.
        logger_name: Name of the logger. Default is "curobo".
    """
    logger = logging.getLogger(logger_name)
    logger.warning(txt, *args, **kwargs)


def log_info(txt: str, logger_name: str = "curobo", *args, **kwargs):
    """Log info message. Also see :py:meth:`logging.Logger.info`.

    Args:
        txt: Info message.
        logger_name: Name of the logger. Default is "curobo".
    """
    logger = logging.getLogger(logger_name)
    logger.info(txt, *args, **kwargs)


def log_error(
    txt: str,
    logger_name: str = "curobo",
    exc_info=True,
    stack_info=False,
    stacklevel: int = 2,
    *args,
    **kwargs
):
    """Log error and raise ValueError.

    Args:
        txt: Helpful message that conveys the error.
        logger_name: Name of the logger. Default is "curobo".
        exc_info: Add exception info to message. See :py:meth:`logging.Logger.error`.
        stack_info: Add stacktracke to message. See :py:meth:`logging.Logger.error`.
        stacklevel: See :py:meth:`logging.Logger.error`. Default value of 2 removes this function
            from the stack trace.

    Raises:
        ValueError: Error message with exception.
    """
    logger = logging.getLogger(logger_name)
    if sys.version_info.major == 3 and sys.version_info.minor <= 7:
        logger.error(txt, exc_info=exc_info, stack_info=stack_info, *args, **kwargs)
    else:
        logger.error(
            txt, exc_info=exc_info, stack_info=stack_info, stacklevel=stacklevel, *args, **kwargs
        )
    raise ValueError(txt)
