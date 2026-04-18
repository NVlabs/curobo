# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Kernel cache for runtime CUDA kernel compilation using cuda.core.

This module provides the infrastructure for compiling and caching CUDA kernels
at runtime, following the pattern from cuda-python examples.
"""

# Standard Library
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from cuda import pathfinder

from curobo._src.runtime import debug_cuda_compile as cuda_debug_compile
from curobo._src.util.logging import log_and_raise, log_debug, log_info, log_warn


def get_cuda_home() -> Optional[str]:
    """Get CUDA installation directory from environment variables.

    Returns:
        Path to CUDA installation or None if not found
    """
    cuda_home = pathfinder.find_nvidia_header_directory("nvrtc")
    return cuda_home



class CudaCoreKernelCache:
    """Cache for compiled CUDA kernels using cuda.core.

    This class handles:
    - Runtime compilation of CUDA kernels via cuda.core
    - Caching compiled kernels based on source content and compilation options
    - Management of CUDA device and architecture
    - Stream wrapper creation for PyTorch integration
    """

    def __init__(self):
        self.compiled_kernels: Dict[str, any] = {}
        self.device: Optional[any] = None
        self.arch: Optional[str] = None

    def initialize(self):
        """Initialize CUDA device and get architecture"""
        if self.device is None:
            try:
                from cuda.core import Device

                self.device = Device()
                self.device.set_current()
                self.arch = f"sm_{self.device.arch}"
                log_info(f"cuda.core backend initialized: {self.arch}")
            except ImportError:
                log_and_raise("cuda.core not available, cannot initialize CudaCoreKernelCache")

    def get_stream_wrapper(self, torch_stream):
        """Create cuda.core compatible stream from PyTorch stream.

        Args:
            torch_stream: PyTorch CUDA stream

        Returns:
            cuda.core Stream object
        """

        class PyTorchStreamWrapper:
            def __init__(self, pt_stream):
                self.pt_stream = pt_stream

            def __cuda_stream__(self):
                stream_id = self.pt_stream.cuda_stream
                return (0, stream_id)

        if self.device is None:
            self.initialize()

        # This doesn't create a new stream, it just wraps the PyTorch stream in a cuda.core
        # Stream object
        cuda_core_stream = self.device.create_stream(PyTorchStreamWrapper(torch_stream))
        return cuda_core_stream

    def get_kernel_hash(
        self, source_files: List[Path], kernel_name: str, compile_flags: List[str]
    ) -> str:
        """Generate cache key based on file contents and compilation options.

        Args:
            source_files: List of kernel source files
            kernel_name: Name of the kernel function
            compile_flags: Compilation flags

        Returns:
            SHA256 hash string for cache key
        """
        content_hash = hashlib.sha256()

        # Hash file contents
        for f in source_files:
            if f.exists():
                content_hash.update(f.read_bytes())
            else:
                log_warn(f"Source file not found: {f}")

        # Hash kernel name and compile flags
        content_hash.update(kernel_name.encode())
        content_hash.update("_".join(compile_flags).encode())

        # Include GPU architecture in hash
        if self.arch:
            content_hash.update(self.arch.encode())

        return content_hash.hexdigest()

    def get_or_compile_kernel(
        self,
        source_files: List[Path],
        kernel_name: str,
        include_dirs: List[Path],
        compile_flags: List[str],
    ):
        """Get cached kernel or compile if not present.

        Args:
            source_files: List of .cu/.cuh files to compile
            kernel_name: Name of kernel function (with template instantiation)
            include_dirs: Include directories for compilation
            compile_flags: Additional nvcc flags

        Returns:
            Compiled kernel object
        """
        self.initialize()

        cache_key = self.get_kernel_hash(source_files, kernel_name, compile_flags)

        if cache_key in self.compiled_kernels:
            log_debug(f"Cache hit: {kernel_name}")
            return self.compiled_kernels[cache_key]

        log_info(f"Compiling kernel: {kernel_name}")
        kernel = self._compile_kernel(source_files, kernel_name, include_dirs, compile_flags)

        self.compiled_kernels[cache_key] = kernel

        # Sync after get_kernel() to ensure the cubin is fully loaded into the
        # GPU context before returning. get_kernel() internally calls
        # cuModuleLoadData() which loads the binary into the current context.
        # Without this sync, subsequent kernel launches may fail if the module
        # isn't fully loaded yet.
        torch.cuda.synchronize()

        return kernel

    def _compile_kernel(
        self,
        source_files: List[Path],
        kernel_name: str,
        include_dirs: List[Path],
        compile_flags: List[str],
    ):
        """Compile kernel using cuda.core.

        Following the pattern from cuda-python KernelHelper, this method:
        1. Reads and combines source files
        2. Sets up include paths (including CUDA system headers)
        3. Compiles using cuda.core Program API

        Args:
            source_files: Kernel source files
            kernel_name: Fully qualified kernel name with template args
            include_dirs: User-specified include directories
            compile_flags: Additional compilation flags

        Returns:
            Compiled kernel object
        """
        from cuda.core import Program, ProgramOptions

        # Read and combine source files
        source_code = self._read_sources(source_files, include_dirs)

        # Setup include paths following common.py pattern
        include_paths = [str(inc) for inc in include_dirs]

        # Add CUDA system includes
        cuda_include = pathfinder.find_nvidia_header_directory("nvrtc")
        if os.path.isdir(cuda_include):
            include_paths.append(cuda_include)
        else:
            log_and_raise("CUDA include directory not found, compilation may fail for system headers")




        # Parse compile flags and map to ProgramOptions parameters
        # Default options from setup.py flags

        options_kwargs = {
            "std": "c++17",
            "arch": self.arch,
            "include_path": include_paths,
            "ftz": True,  # --ftz=true
            "fma": True,  # --fmad=true
            "prec_div": False,  # --prec-div=false
            "prec_sqrt": False,  # --prec-sqrt=false
            "lineinfo": True,  # --generate-line-info
            "device_code_optimize": True,
        }
        if cuda_debug_compile:
            options_kwargs["debug"] = True
            del options_kwargs["device_code_optimize"]

        # Create program options
        options = ProgramOptions(**options_kwargs)

        # Compile
        try:
            prog = Program(source_code, code_type="c++", options=options)
            mod = prog.compile("cubin", name_expressions=(kernel_name,))
            kernel = mod.get_kernel(kernel_name)
            log_info(f"Successfully compiled kernel: {kernel_name}")
            return kernel
        except Exception as e:
            log_and_raise(f"Failed to compile kernel {kernel_name}: {e}")
            raise

    def _read_sources(self, source_files: List[Path], include_dirs: List[Path]) -> str:
        """Read and combine source files.

        Note: This is a simple implementation that reads files sequentially.
        For full #include resolution, a more sophisticated parser would be needed.

        Args:
            source_files: List of kernel source files to read
            include_dirs: Include directories (for reference, not used in simple impl)

        Returns:
            Combined source code string
        """
        # Define standard types that NVRTC doesn't provide by default
        source = """
// Define standard integer types for NVRTC
typedef signed char        int8_t;
typedef short              int16_t;
typedef int                int32_t;
typedef long long          int64_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
#define CUDA_CORE_COMPILE true
"""

        for f in source_files:
            if f.exists():
                source += f"\n// ===== {f.name} =====\n"
                source += f.read_text()
            else:
                log_and_raise(f"Source file not found: {f}")
                raise FileNotFoundError(f"Kernel source file not found: {f}")

        return source
