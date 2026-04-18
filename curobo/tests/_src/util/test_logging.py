# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# Standard Library
import logging

# Third Party
import pytest

# CuRobo
from curobo._src.util.logging import (
    deprecated,
    log_and_raise,
    log_debug,
    log_info,
    log_warn,
    setup_curobo_logger,
    setup_logger,
)


class TestSetupLogger:
    def test_setup_logger_info(self):
        setup_logger(level="info", logger_name="test_logger")
        logger = logging.getLogger("test_logger")
        assert logger.level == logging.INFO

    def test_setup_logger_debug(self):
        setup_logger(level="debug", logger_name="test_logger_debug")
        logger = logging.getLogger("test_logger_debug")
        assert logger.level == logging.DEBUG

    def test_setup_logger_warning(self):
        setup_logger(level="warning", logger_name="test_logger_warn")
        logger = logging.getLogger("test_logger_warn")
        assert logger.level == logging.WARN

    def test_setup_logger_error(self):
        setup_logger(level="error", logger_name="test_logger_error")
        logger = logging.getLogger("test_logger_error")
        assert logger.level == logging.ERROR

    def test_setup_logger_invalid_level(self):
        with pytest.raises(ValueError):
            setup_logger(level="invalid", logger_name="test_logger")

    def test_setup_curobo_logger(self):
        setup_curobo_logger(level="info")
        logger = logging.getLogger("curobo")
        assert logger.level == logging.INFO


class TestLoggingFunctions:
    def test_log_warn(self, caplog):
        with caplog.at_level(logging.WARNING):
            log_warn("Test warning message", logger_name="test_warn_logger")
        assert "Test warning message" in caplog.text

    def test_log_info(self, caplog):
        with caplog.at_level(logging.INFO):
            log_info("Test info message", logger_name="test_info_logger")
        assert "Test info message" in caplog.text

    def test_log_debug(self, caplog):
        with caplog.at_level(logging.DEBUG):
            log_debug("Test debug message", logger_name="test_debug_logger")
        assert "Test debug message" in caplog.text

    def test_log_and_raise(self):
        with pytest.raises(ValueError, match="Test error message"):
            log_and_raise("Test error message", logger_name="test_error_logger")


class TestDeprecated:
    def test_deprecated_decorator(self, caplog):
        @deprecated("Use new_function instead")
        def old_function():
            return "old"

        with caplog.at_level(logging.WARNING):
            result = old_function()
        assert result == "old"
        assert "DEPRECATED" in caplog.text
        assert "old_function" in caplog.text

    def test_deprecated_attribute(self):
        @deprecated("Use new_function instead")
        def old_function():
            return "old"

        assert hasattr(old_function, "__deprecated__")
        assert old_function.__deprecated__ is True

