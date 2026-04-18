/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuda.h>
#include <torch/extension.h>
#include <vector>
#include <array>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

#include <assert.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <math.h>
#include <common/torch_cuda_utils.h>
#include "lbfgs_step_kernel.cuh"

namespace curobo{
  namespace optimization{





// ============================================================================
// LAUNCH HELPER FUNCTIONS AND STRUCTS
// ============================================================================

// Helper struct to hold launch configuration
struct LBFGSLaunchCfg {
    int threadsPerBlock;
    int blocksPerGrid;
    int shared_memory_size;
    int max_shared_memory;
    bool use_shared_buffers;
    bool shared_memory_configured;
};

// ============================================================================
// ENHANCED KERNEL CONFIGURATION AND SELECTION (Design 3 Improved)
// ============================================================================

// Centralized kernel configuration with compile-time validation
struct LBFGSKernelConfig {
    // Single source of truth for supported history_m values
    static constexpr std::array<int, 8> SUPPORTED_HISTORY_M = {5, 6, 7, 15, 24, 27, 28, 31};
    static constexpr int MAX_HISTORY_M = 31;
    static constexpr int MIN_HISTORY_M = 5;

    // Compile-time validation - explicit for static_assert compatibility
    static constexpr bool is_supported(int m) noexcept {
        return m == 5 || m == 6 || m == 7 || m == 15 ||
               m == 24 || m == 27 || m == 28 || m == 31;
    }

    // Runtime validation with detailed error info
    static bool validate_history_m(int m) {
        if (m < MIN_HISTORY_M || m > MAX_HISTORY_M) {
            return false;
        }
        return is_supported(m);
    }
};

// Strongly-typed kernel pair with metadata
template<typename ScalarType, bool rolled_ys>
struct LBFGSKernelPair {
    using SharedKernel = void(*)(ScalarType*, ScalarType*, ScalarType*, ScalarType*,
                                ScalarType*, ScalarType*, ScalarType*, const ScalarType*,
                                float, int, int, int, bool);
    using StableKernel = SharedKernel;  // Same signature

    SharedKernel shared_memory_kernel;
    StableKernel stable_kernel;
    int specialized_for_history_m;  // -1 for runtime kernel
    bool is_specialized;

    constexpr LBFGSKernelPair(SharedKernel shared, StableKernel stable, int history_m = -1)
        : shared_memory_kernel(shared), stable_kernel(stable),
          specialized_for_history_m(history_m), is_specialized(history_m > 0) {}
};

// Helper function to calculate shared memory configuration
inline LBFGSLaunchCfg calculate_lbfgs_launch_config(
    int batch_size, int v_dim, int history_m, bool use_shared_buffers)
{
    LBFGSLaunchCfg config;

    // Basic configuration
    config.threadsPerBlock = v_dim;
    config.blocksPerGrid = batch_size;

    // Calculate shared memory requirements
    const int basic_smem_size = history_m * sizeof(float);
    const int shared_buffer_smem_size = (((2 * v_dim) + 2) * history_m + 32 + 1) * sizeof(float);

    // Shared memory limits
    const int max_shared_base = 48000;
    const int max_shared_allowed = 65536; // Turing limit

    config.max_shared_memory = max_shared_base;
    config.use_shared_buffers = false;
    config.shared_memory_configured = false;

    if (use_shared_buffers) {
        config.shared_memory_size = shared_buffer_smem_size;

        // Check if we can use shared buffers
        if (curobo::common::isVoltaPlus &&
            shared_buffer_smem_size > max_shared_base &&
            shared_buffer_smem_size <= max_shared_allowed) {

            config.max_shared_memory = shared_buffer_smem_size;
            config.use_shared_buffers = true;
        } else if (shared_buffer_smem_size <= max_shared_base) {
            config.use_shared_buffers = true;
        }
    } else {
        config.shared_memory_size = basic_smem_size;
        config.use_shared_buffers = false;
    }

    return config;
}

// Helper function to configure shared memory for a kernel
template<typename KernelType>
inline bool configure_shared_memory(KernelType kernel, LBFGSLaunchCfg& config) {
    if (!curobo::common::isVoltaPlus ||
        config.max_shared_memory <= 48000 ||
        !config.use_shared_buffers) {
        return true;
    }

    cudaError_t result = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, config.max_shared_memory);

    if (result != cudaSuccess) {
        config.max_shared_memory = 48000;
        config.use_shared_buffers = (config.shared_memory_size <= config.max_shared_memory);
        return false;
    }

    config.shared_memory_configured = true;
    return true;
}

// Enhanced kernel factory with better interface and validation
template<typename ScalarType, bool rolled_ys>
class LBFGSKernelFactory {
public:
    using KernelPair = LBFGSKernelPair<ScalarType, rolled_ys>;

    // Main selection interface - cleaner and more explicit
    static constexpr KernelPair select_optimal_kernels(int history_m, bool prefer_specialized = true) {
        if (!prefer_specialized || !LBFGSKernelConfig::is_supported(history_m)) {
            return create_runtime_kernels();
        }

        return create_specialized_kernels(history_m);
    }

private:
    // Specialized kernel creation - constexpr for each supported value
    static constexpr KernelPair create_specialized_kernels(int history_m) {
        // Clean switch with compile-time kernel creation
        switch (history_m) {
            case 5:  return make_specialized_pair<5>();
            case 6:  return make_specialized_pair<6>();
            case 7:  return make_specialized_pair<7>();
            case 15: return make_specialized_pair<15>();
            case 24: return make_specialized_pair<24>();
            case 27: return make_specialized_pair<27>();
            case 28: return make_specialized_pair<28>();
            case 31: return make_specialized_pair<31>();
            default: return create_runtime_kernels();  // Fallback
        }
    }

    // Template helper - compile-time kernel pair creation
    template<int M>
    static constexpr KernelPair make_specialized_pair() {
        static_assert(M > 0, "Specialized kernel requires positive history_m");
        static_assert(LBFGSKernelConfig::is_supported(M),
                     "history_m not in supported values list");

        return KernelPair{
            kernel_lbfgs_step_shared_memory<ScalarType, rolled_ys, M>,
            kernel_lbfgs_step<ScalarType, rolled_ys, M>,
            M
        };
    }

    // Runtime kernel creation
    static constexpr KernelPair create_runtime_kernels() {
        return KernelPair{
            kernel_lbfgs_step_shared_memory<ScalarType, rolled_ys>,
            kernel_lbfgs_step<ScalarType, rolled_ys>,
            -1  // Indicates runtime kernel
        };
    }
};

// Clean, modern interface for kernel selection
template<typename ScalarType = float, bool rolled_ys = false>
constexpr auto select_lbfgs_kernels(int history_m, bool use_fixed_m = true) {
    using Factory = LBFGSKernelFactory<ScalarType, rolled_ys>;

    // Self-documenting parameter names and behavior
    return Factory::select_optimal_kernels(history_m, use_fixed_m);
}

// Convenience function with default template parameters
constexpr auto select_lbfgs_kernels_float(int history_m, bool use_fixed_m = true) {
    return select_lbfgs_kernels<float, false>(history_m, use_fixed_m);
}

// ============================================================================
// SIMPLIFIED LAUNCH FUNCTION
// ============================================================================

std::vector<torch::Tensor>
launch_lbfgs_step(torch::Tensor step_vec, torch::Tensor rho_buffer,
                torch::Tensor y_buffer, torch::Tensor s_buffer, torch::Tensor q,
                torch::Tensor grad_q, torch::Tensor x_0, torch::Tensor grad_0,
                const float epsilon, const int batch_size, const int history_m,
                const int v_dim, const bool stable_mode, const bool use_shared_buffers)
{
    // Validate all inputs
    curobo::common::validate_cuda_input(step_vec, "step_vec");
    curobo::common::validate_cuda_input(rho_buffer, "rho_buffer");
    curobo::common::validate_cuda_input(y_buffer, "y_buffer");
    curobo::common::validate_cuda_input(s_buffer, "s_buffer");
    curobo::common::validate_cuda_input(q, "q");
    curobo::common::validate_cuda_input(x_0, "x_0");
    curobo::common::validate_cuda_input(grad_0, "grad_0");
    curobo::common::validate_cuda_input(grad_q, "grad_q");

    // Basic validation
    assert(v_dim < 1024 && history_m < 32);

    // Calculate launch configuration
    auto config = calculate_lbfgs_launch_config(batch_size, v_dim, history_m, use_shared_buffers);

        // Select appropriate kernels using enhanced interface
    const bool use_fixed_m = true;
    auto kernel_pair = select_lbfgs_kernels_float(history_m, use_fixed_m);

    // Configure shared memory if needed
    bool shared_configured = configure_shared_memory(kernel_pair.shared_memory_kernel, config);

    // Get CUDA stream
    cudaStream_t stream = curobo::common::get_cuda_stream();

    // Launch appropriate kernel based on configuration
    if (config.use_shared_buffers && config.shared_memory_size <= config.max_shared_memory) {
        kernel_pair.shared_memory_kernel<<<config.blocksPerGrid, config.threadsPerBlock, config.shared_memory_size, stream>>>(
            step_vec.data_ptr<float>(),
            rho_buffer.data_ptr<float>(),
            y_buffer.data_ptr<float>(),
            s_buffer.data_ptr<float>(),
            q.data_ptr<float>(),
            x_0.data_ptr<float>(),
            grad_0.data_ptr<float>(),
            grad_q.data_ptr<float>(),
            epsilon, batch_size, history_m, v_dim, stable_mode);
    } else {
        const int basic_smem_size = history_m * v_dim * sizeof(float);
        kernel_pair.stable_kernel<<<config.blocksPerGrid, config.threadsPerBlock, basic_smem_size, stream>>>(
            step_vec.data_ptr<float>(),
            rho_buffer.data_ptr<float>(),
            y_buffer.data_ptr<float>(),
            s_buffer.data_ptr<float>(),
            q.data_ptr<float>(),
            x_0.data_ptr<float>(),
            grad_0.data_ptr<float>(),
            grad_q.data_ptr<float>(),
            epsilon, batch_size, history_m, v_dim, stable_mode);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return { step_vec, rho_buffer, y_buffer, s_buffer, x_0, grad_0 };
}
}
}
