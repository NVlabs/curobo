# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Third Party
import torch

# CuRobo
from curobo._src.graph_planner.graph_planner_prm_cfg import PRMGraphPlannerCfg
from curobo._src.util.logging import log_and_raise
from curobo._src.util.sampling.sample_buffer import SampleBuffer
from curobo._src.util.torch_util import get_torch_jit_decorator


class NodeSamplingStrategy:
    """Class responsible for sampling robot configurations for graph planning.

    This class encapsulates all sampling-related functionality including uniform sampling,
    sampling within ellipsoids, and collision checking to find feasible samples.
    """

    def __init__(
        self,
        config: PRMGraphPlannerCfg,
        action_lower_bounds: torch.Tensor,
        action_upper_bounds: torch.Tensor,
        cspace_distance_weight: torch.Tensor,
        action_dim: int,
        check_feasibility_fn,
        device_cfg=None,
    ):
        """Initialize the sampling strategy.

        Args:
            config: Configuration for the PRM planner
            action_lower_bounds: Lower bounds for the action space
            action_upper_bounds: Upper bounds for the action space
            cspace_distance_weight: Weights for distance calculation in configuration space
            action_dim: Dimensionality of the action space (robot DOF)
            check_feasibility_fn: Function to check if a configuration is feasible
            device_cfg: Tensor device and type arguments
        """
        self.config = config
        self.device_cfg = config.device_cfg if device_cfg is None else device_cfg
        self.action_bound_lows = action_lower_bounds
        self.action_bound_highs = action_upper_bounds
        self.cspace_distance_weight = cspace_distance_weight
        self.action_dim = action_dim
        self.check_feasibility_fn = check_feasibility_fn

        # Initialize the Halton sequence generator for sampling
        self.action_sample_generator = SampleBuffer.create_halton_sample_buffer(
            ndims=self.action_dim,
            device_cfg=self.device_cfg,
            up_bounds=self.action_bound_highs,
            low_bounds=self.action_bound_lows,
            seed=self.config.sampler_seed,
        )

        # Create a rotation frame tensor for ellipsoidal sampling
        self._action_dim_rot_frame = torch.eye(
            self.action_dim, device=self.device_cfg.device, dtype=self.device_cfg.dtype
        )

    def generate_action_samples(
        self, n_samples: int, bounded: bool = True, unit_ball: bool = False
    ):
        """Generate action samples using the Halton sequence.

        Args:
            n_samples: Number of samples to generate
            bounded: Whether to bound samples to joint limits
            unit_ball: Whether to generate samples on a unit ball

        Returns:
            Tensor of action samples (batch_size, action_dim)
        """
        if unit_ball:
            # Generate samples from a Gaussian distribution
            halton_samples = self.action_sample_generator.get_gaussian_samples(
                n_samples, variance=1.0
            )
            # Normalize to place on unit sphere
            halton_samples = halton_samples / torch.norm(halton_samples, dim=-1, keepdim=True)

            if self.action_dim < 3:
                # For lower dimensions, create a filled circle rather than just the circumference
                radius_samples = self.action_sample_generator.get_samples(n_samples, bounded=False)
                radius_samples = torch.clamp(radius_samples[:, 0:1], 0.0, 1.0)
                halton_samples = radius_samples * halton_samples
        else:
            # Regular bounded/unbounded samples
            halton_samples = self.action_sample_generator.get_samples(n_samples, bounded=bounded)

        return halton_samples

    def check_samples_feasibility(self, action_samples):
        """Check if action samples are collision-free.

        Args:
            action_samples: Tensor of action samples to check

        Returns:
            Boolean mask indicating which samples are feasible
        """
        if len(action_samples.shape) != 2:
            log_and_raise("action_samples must be a 2D tensor (batch_size, action_dim)")

        # Use the provided collision check function
        return self.check_feasibility_fn(action_samples)

    def get_feasible_sample_set(self, x_samples):
        """Filter out samples that are in collision.

        Args:
            x_samples: Tensor of action samples

        Returns:
            Tensor of collision-free samples
        """
        mask = self.check_samples_feasibility(x_samples)
        x_samples = x_samples[mask]
        return x_samples

    def generate_feasible_action_samples(self, num_samples: int):
        """Generate feasible action samples.

        Args:
            num_samples: Number of feasible samples to generate

        Returns:
            Tensor of feasible action samples
        """
        action_samples = self.generate_action_samples(
            n_samples=num_samples + int(num_samples * self.config.sample_rejection_ratio),
            bounded=True,
        )
        feasible_action_samples = self.get_feasible_sample_set(action_samples)
        if feasible_action_samples.shape[0] < num_samples:
            return feasible_action_samples
        else:
            return feasible_action_samples[:num_samples]

    def generate_feasible_samples(self, num_samples: int) -> torch.Tensor:
        """Generate feasible samples for PRM roadmap.

        Args:
            num_samples: Number of feasible samples to generate

        Returns:
            Tensor of feasible samples
        """
        # Generate new samples:
        x_samples = self.generate_action_samples(
            n_samples=num_samples + int(num_samples * self.config.sample_rejection_ratio),
            bounded=True,
        )
        x_search = self.get_feasible_sample_set(x_samples)
        if x_search.shape[0] < num_samples:
            return x_search
        else:
            return x_search[:num_samples]

    # @profiler.record_function("sampling_strategy/generate_feasible_samples_in_ellipsoid")
    def generate_feasible_samples_in_ellipsoid(
        self,
        x_start: torch.Tensor,
        x_goal: torch.Tensor,
        num_samples: int,
        max_sampling_radius: torch.Tensor,
    ) -> torch.Tensor:
        """Generate feasible samples within an ellipsoid defined by start and goal configurations.

        Args:
            x_start: Start configuration
            x_goal: Goal configuration
            num_samples: Number of samples to generate
            max_sampling_radius: Maximum radius for sampling

        Returns:
            Tensor of feasible samples within the ellipsoid
        """
        if len(x_start.shape) != 1:
            log_and_raise("x_start must be a 1D tensor (action_dim)")
        if len(x_goal.shape) != 1:
            log_and_raise("x_goal must be a 1D tensor (action_dim)")
        if x_start.shape[0] != self.action_dim:
            log_and_raise("x_start must have the same number of dimensions as the action space")
        if x_goal.shape[0] != self.action_dim:
            log_and_raise("x_goal must have the same number of dimensions as the action space")
        unit_ball_samples = self.generate_action_samples(
            n_samples=num_samples + int(num_samples * self.config.sample_rejection_ratio),
            unit_ball=True,
        )

        # compute cost_to_go:
        if self.config.ellipsoid_projection_method == "householder":
            x_samples = self.jit_transform_unit_ball_to_ellipsoid_householder(
                x_start,
                x_goal,
                self.cspace_distance_weight,
                max_sampling_radius,
                self.action_dim,
                self._action_dim_rot_frame,
                unit_ball_samples,
                self.action_bound_lows,
                self.action_bound_highs,
            )
        elif self.config.ellipsoid_projection_method == "svd":
            x_samples = self.jit_transform_unit_ball_to_ellipsoid_svd(
                x_start,
                x_goal,
                self.cspace_distance_weight,
                max_sampling_radius,
                self.action_dim,
                self._action_dim_rot_frame,
                unit_ball_samples,
                self.action_bound_lows,
                self.action_bound_highs,
            )
        elif self.config.ellipsoid_projection_method == "approximate":
            x_samples = self.jit_transform_unit_ball_to_ellipsoid_approximate(
                x_start,
                x_goal,
                self.cspace_distance_weight,
                max_sampling_radius,
                self.action_dim,
                self._action_dim_rot_frame,
                unit_ball_samples,
                self.action_bound_lows,
                self.action_bound_highs,
            )
        x_search = self.get_feasible_sample_set(x_samples)

        # x_search could be less than num_nodes_to_sample_per_iter if all get rejected.
        # NOTE: What to do in this case?
        if x_search.shape[0] < num_samples:
            return x_search
        else:
            return x_search[:num_samples]

    def _validate_ellipsoid_inputs(self, x_start: torch.Tensor, x_goal: torch.Tensor):
        """Validate inputs for ellipsoidal sampling.

        Args:
            x_start: Start configuration
            x_goal: Goal configuration

        Raises:
            ValueError: If inputs have incorrect shapes
        """
        if len(x_start.shape) != 1:
            log_and_raise("x_start must be a 1D tensor (action_dim)")
        if len(x_goal.shape) != 1:
            log_and_raise("x_goal must be a 1D tensor (action_dim)")
        if x_start.shape[0] != self.action_dim:
            log_and_raise("x_start must have the same number of dimensions as the action space")
        if x_goal.shape[0] != self.action_dim:
            log_and_raise("x_goal must have the same number of dimensions as the action space")

    def compute_distance_from_line(
        self,
        vertices: torch.Tensor,
        x_start: torch.Tensor,
        x_goal: torch.Tensor,
    ):
        """Compute the distance from vertices to a line defined by start and goal.

        Args:
            vertices: Batch of vertices
            x_start: Start configuration
            x_goal: Goal configuration

        Returns:
            Distances from vertices to the line
        """
        self._validate_line_distance_inputs(vertices, x_start, x_goal)
        return self.jit_compute_distance_from_line(vertices, x_start, x_goal)

    def _validate_line_distance_inputs(
        self, vertices: torch.Tensor, x_start: torch.Tensor, x_goal: torch.Tensor
    ):
        """Validate inputs for line distance computation.

        Args:
            vertices: Batch of vertices
            x_start: Start point of the line segment
            x_goal: End point of the line segment

        Raises:
            ValueError: If inputs have incorrect shapes
        """
        if len(vertices.shape) != 2:
            log_and_raise("vertices must be a 2D tensor (batch_size, action_dim)")
        if len(x_start.shape) != 1:
            log_and_raise("x_start must be a 1D tensor (action_dim)")
        if len(x_goal.shape) != 1:
            log_and_raise("x_goal must be a 1D tensor (action_dim)")
        if x_start.shape[0] != self.action_dim:
            log_and_raise("x_start must have the same number of dimensions as the action space")
        if x_goal.shape[0] != self.action_dim:
            log_and_raise("x_goal must have the same number of dimensions as the action space")

    def reset_seed(self):
        """Reset the seed of the Halton sequence generator."""
        self.action_sample_generator.reset()

    @staticmethod
    @get_torch_jit_decorator(dynamic=True, slow_to_compile=True)
    def jit_transform_unit_ball_to_ellipsoid_householder(
        x_start,
        x_goal,
        distance_weight,
        max_sampling_radius: torch.Tensor,
        action_dim: int,
        rot_frame_col: torch.Tensor,
        unit_ball_samples: torch.Tensor,
        low_bounds: torch.Tensor,
        high_bounds: torch.Tensor,
    ) -> torch.Tensor:
        """Transform unit ball samples to an ellipsoid using Householder transformation.

        This function transforms samples from a unit ball to an ellipsoid defined by
        start and goal configurations, with the major axis aligned along the direction
        from start to goal.

        Args:
            x_start: Start configuration
            x_goal: Goal configuration
            distance_weight: Weights for distance calculation
            max_sampling_radius: Maximum radius of the sampling ellipsoid
            action_dim: Dimensionality of the action space
            rot_frame_col: Rotation frame tensor
            unit_ball_samples: Samples from a unit ball
            low_bounds: Lower bounds for the action space
            high_bounds: Upper bounds for the action space

        Returns:
            Tensor of samples transformed to an ellipsoid
        """
        # Calculate normalized direction
        direction = x_goal - x_start
        direction_norm = torch.norm(direction * distance_weight)
        min_sampling_radius = direction_norm
        direction = direction / direction_norm

        # Create first standard basis vector
        e1 = torch.zeros_like(direction)
        e1[0] = 1.0

        # Compute v for Householder transformation
        # We want to reflect e1 to direction or -direction (whichever has positive first component)
        if direction[0] >= 0:
            # Make sure we're transforming to direction, not -direction
            v = direction - e1
        else:
            # If first component is negative, reflect to -direction instead
            v = direction + e1

        # Normalize v
        v_norm = torch.norm(v)
        if v_norm > 1e-10:  # Avoid division by zero
            v = v / v_norm
        else:
            # If vectors are already aligned, no transformation needed
            v = torch.zeros_like(direction)

        # Create Householder matrix: I - 2 * outer(v, v)
        H = rot_frame_col
        H = H - 2.0 * torch.outer(v, v)

        # H is now our rotation matrix
        C = H

        # Create scaling matrix with different scales for major and minor axes
        scale_values = torch.zeros_like(x_start)
        scale_values[0:1] = max_sampling_radius / 2.0
        scale_values[1:] = (max_sampling_radius**2 - min_sampling_radius**2) / 2.0

        # Apply transformation: rotate, scale, and translate
        transform = C @ torch.diag(scale_values)
        x_center = (x_start + x_goal) / 2.0
        x_samples = ((transform @ unit_ball_samples.T).T / distance_weight) + x_center

        # Ensure samples are within joint limits
        x_samples = torch.clamp(x_samples, low_bounds, high_bounds).contiguous()

        return x_samples

    @staticmethod
    # @profiler.record_function("base_graph_planner/jit_transform_unit_ball_to_ellipsoid")
    @get_torch_jit_decorator(dynamic=True, slow_to_compile=True)
    def jit_transform_unit_ball_to_ellipsoid_svd(
        x_start,
        x_goal,
        distance_weight,
        max_sampling_radius: torch.Tensor,
        action_dim: int,
        rot_frame_col: torch.Tensor,
        unit_ball_samples: torch.Tensor,
        low_bounds: torch.Tensor,
        high_bounds: torch.Tensor,
    ) -> torch.Tensor:
        # Calculate normalized direction - we know points are far apart
        direction = x_goal - x_start
        direction_norm = torch.norm(direction * distance_weight)
        min_sampling_radius = direction_norm
        direction = (direction / direction_norm).unsqueeze(1)

        # Generate rotation frame (first column is the direction)
        M = direction @ rot_frame_col[:, 0:1].T

        # Compute SVD only once - this is the most expensive operation
        U, _, V = torch.svd(M, compute_uv=True, some=False)

        # Create orientation correction vector
        # (all ones except last element which preserves orientation)
        vec = torch.ones(action_dim, device=x_start.device, dtype=x_start.dtype)
        vec[-1] = torch.det(U) * torch.det(V)

        # Create rotation matrix
        C = U @ torch.diag(vec) @ V.T

        # Create scaling matrix directly
        # First element is the major axis (along the direction)
        # Other elements are the minor axes
        scale_values = torch.zeros_like(x_start)
        scale_values[0] = max_sampling_radius / 2.0
        scale_values[1:] = (max_sampling_radius**2 - min_sampling_radius**2) / 2.0

        # Precompute the combined transformation matrix
        transform = C @ torch.diag(scale_values)

        # Calculate center point
        x_center = (x_start + x_goal) / 2.0

        # Apply transformation efficiently with batched operations
        # 1. Transform samples: transform @ unit_ball_samples.T
        # 2. Transpose result: .T
        # 3. Apply distance weight normalization: / distance_weight
        # 4. Translate to center: + x_center
        x_samples = ((transform @ unit_ball_samples.T).T / distance_weight) + x_center

        # Ensure samples are within joint limits
        x_samples = torch.clamp(x_samples, low_bounds, high_bounds).contiguous()

        return x_samples

    @staticmethod
    # @profiler.record_function("base_graph_planner/jit_transform_unit_ball_to_ellipsoid")
    @get_torch_jit_decorator(dynamic=True, slow_to_compile=True)
    def jit_transform_unit_ball_to_ellipsoid_approximate(
        x_start,
        x_goal,
        distance_weight,
        max_sampling_radius: torch.Tensor,
        action_dim: int,
        rot_frame_col: torch.Tensor,
        unit_ball_samples: torch.Tensor,
        low_bounds: torch.Tensor,
        high_bounds: torch.Tensor,
    ) -> torch.Tensor:
        """Very approximate version"""
        # Calculate direction and center
        x_center = (x_start + x_goal) / 2.0
        direction = x_goal - x_start
        direction_norm = torch.norm(direction * distance_weight)
        min_sampling_radius = direction_norm
        direction = direction / direction_norm

        # Decompose samples
        # Project onto direction vector to get component along major axis
        proj = torch.matmul(unit_ball_samples, direction)

        # Perpendicular components
        perp_components = unit_ball_samples - proj.unsqueeze(1) * direction
        perp_norms = torch.norm(perp_components, dim=1, keepdim=True)
        perp_normalized = torch.where(
            perp_norms > 1e-10, perp_components / perp_norms, torch.zeros_like(perp_components)
        )

        # Scale components differently
        major_component = proj.unsqueeze(1) * direction * max_sampling_radius
        minor_component = perp_normalized * perp_norms * min_sampling_radius

        # Combine and translate
        x_samples = major_component + minor_component + x_center

        # Clamp to joint limits
        return torch.clamp(x_samples, low_bounds, high_bounds)

    @staticmethod
    @get_torch_jit_decorator(dynamic=True, slow_to_compile=True)
    def jit_compute_distance_from_line(
        vertices: torch.Tensor, x_start: torch.Tensor, x_goal: torch.Tensor
    ):
        """Compute the distance from vertices to a line segment.

        Args:
            vertices: Batch of vertices
            x_start: Start point of the line segment
            x_goal: End point of the line segment

        Returns:
            Distances from vertices to the line segment
        """
        x_start_expanded = x_start.unsqueeze(0)

        # Vector from start to goal
        line_vector = x_goal - x_start
        line_length = torch.norm(line_vector)

        # Vector from start to each vertex
        vertex_vectors = vertices - x_start_expanded

        # Project vertex vectors onto line vector
        projections = torch.sum(vertex_vectors * line_vector, dim=1) / (line_length * line_length)

        # Clamp projections to [0,1] to get points on line segment
        projections = torch.clamp(projections, 0.0, 1.0)

        # Get points on line segment
        line_points = x_start_expanded + projections.unsqueeze(1) * line_vector

        # Calculate distances from vertices to line points
        distances = torch.norm(vertices - line_points, dim=1)
        return distances
