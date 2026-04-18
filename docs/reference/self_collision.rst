Robot Self-Collision
====================================

Self-collision avoidance is a required behavior for robot motion generation
(IK, MPC, motion planning). While single end-effector manipulators are usually designed to have
limited self-collisions by limiting joint limits or with specific geometric shapes, multi-arm and
humanoid robots will have a higher likelihood of self-collisions. It is therefore required to incorporate
self-collision avoidance within the optimization process during IK, MPC, and motion planning.

cuRobo represents the robot with spheres. This leads to a larger number of
self-collision checks than if the robot was represented with meshes, as each mesh is represented with
many spheres (20+). During self-collision checks, the distance needs to be computed between every
sphere of each link to every sphere of other links. However, sphere-sphere collision checks are
much cheaper than mesh-mesh collision checks, and we compute them in parallel leveraging
the many cores of a GPU.

When calculating self-collisions, we also need to consider two cases:

1. Some link pairs will always be in collision and are safe to be in collision. E.g., two links attached with a joint (most consecutive links).

2. Some link pairs will never be in collision due to the mechanical design of the robot. We can skip calculating collisions for these pairs.

cuRobo uses a configuration file to specify which link pairs can be ignored for self-collision checks.

1. ``self_collision_ignore``: A dictionary that maps each link to the other links to skip during self-collision checks.


cuRobo uses a single max-reduction kernel for self-collision cost. It returns
the largest penetration distance across all enabled sphere pairs; this is the
scalar minimized during IK, MPC, and motion planning.

The same kernel can optionally write every pair's signed distance to a per-pair
buffer by setting ``store_pair_distance=True`` on
:class:`~curobo._src.cost.cost_self_collision_cfg.SelfCollisionCostCfg`. This is
slower (extra global-memory writes and a larger output buffer) but useful for:

- **Debugging misconfigurations**, by inspecting which specific sphere pairs
  are in collision for a given joint configuration.
- **Generating the** ``self_collision_ignore`` **dictionary automatically**, by
  sampling collision-free joint configurations and recording pairs that are
  never in collision across the sample.

The self-collision parameters (collision pair indices, sphere padding, block
partitioning) are generated and stored in
:class:`~curobo.robot.SelfCollisionKinematicsCfg`.

cuRobo precomputes the index pairs of spheres to check for self-collisions, taking into account the ``self_collision_ignore`` dictionary. This also gives us the total number of sphere pairs to check. We also allow for padding each link's spheres with a radius buffer as we want to be
conservative in our self-collision checks. This is stored per sphere as the self-collision method has no knowledge of which sphere maps to which link.

For larger robots (e.g., bimanual or humanoid, ~1000 spheres), cuRobo splits the
reduction across multiple thread blocks using a two-kernel design: the first
kernel computes a per-block max, and a second kernel performs a warp-level
reduction over the per-block results. Implementation details live in
``curobo/_src/curobolib/kernels/geometry/self_collision/``.


Calculating Self-Collision Skips
--------------------------------

Given robot spheres, cuRobo adds a small padding (``self_collision_buffer``, 2 cm
by default) to each sphere. For a joint configuration that is known to be
collision-free, the max-reduction kernel (run with ``store_pair_distance=True``)
returns every pair's signed distance; any pair reported as in collision is added
to ``self_collision_ignore``.

To skip pairs that are never in collision, cuRobo samples ``n_samples`` joint
configurations uniformly within joint limits and records per-pair distances
across the sample. Pairs that are never in collision across all samples are
added to ``self_collision_ignore``.
