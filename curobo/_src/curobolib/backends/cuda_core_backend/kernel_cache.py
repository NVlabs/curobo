# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Kernel cache for runtime CUDA kernel compilation using cuda.core.

This module provides the infrastructure for compiling and caching CUDA kernels
at runtime, following the pattern from cuda-python examples.
"""

# Standard Library
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


class _PyTorchStreamAdapter:
    """Expose an existing PyTorch CUDA stream through the CUDA stream protocol."""

    def __init__(self, stream_handle: int) -> None:
        self._stream_handle = stream_handle

    def __cuda_stream__(self) -> Tuple[int, int]:
        return (0, self._stream_handle)


PreparedKernelKey = Tuple[str, Tuple[str, ...], Optional[str]]
KernelContentKey = Tuple[Tuple[Tuple[str, Optional[bytes]], ...], str, Tuple[str, ...], Optional[str]]
StreamKey = Tuple[Optional[int], int]


class CudaCoreKernelCache:
    """Cache for compiled CUDA kernels using cuda.core.

    This class handles:
    - Runtime compilation of CUDA kernels via cuda.core
    - Caching compiled kernels based on source content and compilation options
    - Management of CUDA device and architecture
    - Stream wrapper creation for PyTorch integration
    """

    def __init__(self) -> None:
        """Initialize empty compiled-kernel, prepared-kernel, and stream caches."""
        self.compiled_kernels: Dict[KernelContentKey, Any] = {}
        self._prepared_kernels: Dict[PreparedKernelKey, Any] = {}
        self._stream_wrappers: Dict[StreamKey, Any] = {}
        self.device: Optional[Any] = None
        self.arch: Optional[str] = None

    def initialize(self) -> None:
        """Initialize the CUDA device and get its architecture."""
        if self.device is None:
            try:
                from cuda.core import Device

                self.device = Device()
                self.device.set_current()
                self.arch = f"sm_{self.device.arch}"
                log_info(f"cuda.core backend initialized: {self.arch}")
            except ImportError:
                log_and_raise("cuda.core not available, cannot initialize CudaCoreKernelCache")

    def get_stream_wrapper(self, torch_stream: torch.cuda.Stream) -> Any:
        """Return a cached cuda.core wrapper for an existing PyTorch CUDA stream.

        Args:
            torch_stream: PyTorch CUDA stream

        Returns:
            cuda.core Stream object
        """
        if self.device is None:
            self.initialize()

        stream_handle = torch_stream.cuda_stream
        stream_key = (torch_stream.device.index, stream_handle)
        cuda_core_stream = self._stream_wrappers.get(stream_key)
        if cuda_core_stream is not None:
            return cuda_core_stream

        # This doesn't create a new native stream. It wraps the PyTorch stream in a
        # cuda.core Stream object so launches remain ordered on the active stream.
        cuda_core_stream = self.device.create_stream(_PyTorchStreamAdapter(stream_handle))
        self._stream_wrappers[stream_key] = cuda_core_stream
        return cuda_core_stream

    def clear(self) -> None:
        """Clear compiled kernels, prepared lookups, and CUDA stream wrappers."""
        self.compiled_kernels.clear()
        self._prepared_kernels.clear()
        self._stream_wrappers.clear()

    def get_kernel_key(
        self, source_files: List[Path], kernel_name: str, compile_flags: List[str]
    ) -> KernelContentKey:
        """Create an exact cache key from source contents and compilation options.

        Args:
            source_files: List of kernel source files
            kernel_name: Name of the kernel function
            compile_flags: Compilation flags

        Returns:
            Source paths and bytes, kernel name, flags, and GPU architecture.
        """
        sources: List[Tuple[str, Optional[bytes]]] = []
        for f in source_files:
            if f.exists():
                sources.append((str(f), f.read_bytes()))
            else:
                log_warn(f"Source file not found: {f}")
                sources.append((str(f), None))
        return (tuple(sources), kernel_name, tuple(compile_flags), self.arch)

    def get_or_compile_kernel(
        self,
        source_files: List[Path],
        kernel_name: str,
        include_dirs: List[Path],
        compile_flags: List[str],
    ) -> Any:
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

        prepared_key = (
            kernel_name,
            tuple(compile_flags),
            self.arch,
        )
        if not cuda_debug_compile:
            prepared_kernel = self._prepared_kernels.get(prepared_key)
            if prepared_kernel is not None:
                return prepared_kernel

        cache_key = self.get_kernel_key(source_files, kernel_name, compile_flags)

        if cache_key in self.compiled_kernels:
            log_debug(f"Cache hit: {kernel_name}")
            kernel = self.compiled_kernels[cache_key]
            self._prepared_kernels[prepared_key] = kernel
            return kernel

        log_info(f"Compiling kernel: {kernel_name}")
        kernel = self._compile_kernel(source_files, kernel_name, include_dirs, compile_flags)

        self.compiled_kernels[cache_key] = kernel
        self._prepared_kernels[prepared_key] = kernel

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

        # Add CUDA system includes. If kernels start including CCCL or
        # cooperative_groups headers again, add "cccl" here.
        for libname in ("cudart", "nvrtc"):
            cuda_include = pathfinder.find_nvidia_header_directory(libname)
            if cuda_include is not None and os.path.isdir(cuda_include):
                include_paths.append(cuda_include)

        if not any(os.path.exists(os.path.join(path, "nvrtc.h")) for path in include_paths):
            log_and_raise("CUDA NVRTC include directory not found, compilation may fail")




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
"""

        for f in source_files:
            if f.exists():
                source += f"\n// ===== {f.name} =====\n"
                source += f.read_text()
            else:
                log_and_raise(f"Source file not found: {f}")
                raise FileNotFoundError(f"Kernel source file not found: {f}")

        return source
