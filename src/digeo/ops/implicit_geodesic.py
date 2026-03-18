import torch
from torch import Tensor
from typing import Tuple, Optional

from digeo.implicit_surface import ImplicitSurface, ImplicitPointBatch


class ImplicitGeodesicInfo:
    def __init__(
        self,
        end_directions: Optional[Tensor] = None,
        rotation: Optional[Tensor] = None,
        debug_tensor: Optional[Tensor] = None,
    ):
        """
        GeodesicInfo equivalent for implicit models.
        """
        self.end_directions = end_directions
        self.rotation_tensor = rotation
        self.debug_tensor = debug_tensor

    @property
    def rotation(self) -> Tensor:
        if self.rotation_tensor is None:
            raise RuntimeError(
                "Rotation tensor is not set. Set 'save_parallel_transport=True' "
                "to save it."
            )
        return self.rotation_tensor

    def transport(self, v: Tensor) -> Tensor:
        if self.rotation_tensor is None:
            raise RuntimeError(
                "Rotation tensor is not set. Set 'save_parallel_transport=True' "
                "to save it."
            )
        # Apply R to row vector: v_out = v @ R^T
        return torch.bmm(v.unsqueeze(1), self.rotation_tensor.transpose(1, 2)).squeeze(1)

    def transport_inv(self, v: Tensor) -> Tensor:
        if self.rotation_tensor is None:
            raise RuntimeError(
                "Rotation tensor is not set. Set 'save_parallel_transport=True' "
                "to save it."
            )
        # Apply R^-1 to row vector: v_out = v @ R
        return torch.bmm(v.unsqueeze(1), self.rotation_tensor).squeeze(1)


def trace_implicit_geodesics(
    surface: ImplicitSurface,
    starts: ImplicitPointBatch,
    dirs: Tensor,
    dt: float = 0.05,
    max_steps: int = 200,
    proj_iters: int = 3,
    save_parallel_transport: bool = False,
    save_end_direction: bool = False,
    debug: bool = False,
) -> Tuple[ImplicitPointBatch, ImplicitGeodesicInfo]:
    """
    Computes straightest geodesics on an implicit surface (e.g., an SDF).

    This solver uses explicit forward Euler integration:
      1. Step along the current tangent direction.
      2. Project the point back onto the zero level-set using Newton-Raphson
         iterations along the gradient of the SDF.
      3. Parallel transport the direction vector onto the new tangent plane.

    Parameters
    ----------
    surface : ImplicitSurface
        The implicit surface.
    starts : ImplicitPointBatch
        The starting coordinates.
    dirs : Tensor
        (B, 3) tensor representing the initial velocity/tangent direction. The magnitude
        of this vector determines the arc length traced.
    dt : float
        Integration step size (pseudo-time).
        The actual step taken in 3D is `dt * dirs`.
    max_steps : int
        The maximum number of integration steps.
        Total length traced per path is `dt * max_steps * ||dirs||`.
    proj_iters : int
        The number of Newton-Raphson projection iterations to use to return
        to the surface after an integration step.
    save_parallel_transport : bool
        If true, computes the full rotation matrix from the start point's tangent
        plane to the end point's tangent plane.
    save_end_direction : bool
        If true, stores the final tangent direction.
    debug : bool
        If true, stores the entire point trajectory.

    Returns
    -------
    ImplicitPointBatch
        The final coordinates on the surface.
    ImplicitGeodesicInfo
        A struct containing rotation matrices, end directions, and debug info.
    """
    if dirs.dim() != 2 or dirs.shape[1] != 3:
        raise RuntimeError("dirs must be of shape (B, 3)")

    B = dirs.shape[0]
    if len(starts.positions) != B:
        raise RuntimeError(f"starts and dirs batch sizes must match. ({len(starts.positions)} vs {B})")

    # Clone initial state
    curr_pts = starts.positions.clone()
    curr_dirs = dirs.clone()

    # Precompute initial normals and ensure starting directions are on the tangent plane
    # If the user provides an off-tangent direction, we project it.
    initial_normals = surface.eval_normal(curr_pts)
    curr_dirs = curr_dirs - torch.sum(curr_dirs * initial_normals, dim=-1, keepdim=True) * initial_normals

    # Store initial state for parallel transport computations
    start_dirs_unit = curr_dirs.clone()
    start_dirs_norm = torch.linalg.norm(start_dirs_unit, dim=-1, keepdim=True)
    idle_idx = start_dirs_norm.squeeze(-1) < 1e-6
    start_dirs_unit = start_dirs_unit / start_dirs_norm.clamp(min=1e-6)

    trajectory = []
    if debug:
        trajectory.append(curr_pts.clone())

    # Calculate step vector lengths.
    step_vecs = curr_dirs * dt

    # We step exactly `max_steps` times. For paths of varying length,
    # users could scale the input `dirs` tensor appropriately, so ||dirs||
    # encodes the path length.
    for step in range(max_steps):
        # 1. Take a step in the tangent space
        next_pts = curr_pts + step_vecs

        # 2. Project back onto the implicit surface (Newton-Raphson)
        for _ in range(proj_iters):
            sdf_val = surface.eval_sdf(next_pts)
            grad_val = surface.eval_normal(next_pts)

            # Newton step: x_{k+1} = x_k - f(x_k) * grad(f(x_k)) / ||grad(f(x_k))||^2
            # Since grad_val is normalized, ||grad_val||^2 is 1.
            # Assuming sdf_val is (B, 1) and grad_val is (B, 3).
            if sdf_val.dim() == 1:
                sdf_val = sdf_val.unsqueeze(-1)

            next_pts = next_pts - sdf_val * grad_val

        # 3. Compute parallel transport
        new_normals = surface.eval_normal(next_pts)

        # To maintain the speed/length of the vector over the curve properly,
        # we transport the direction then normalize to the original speed.
        step_len = torch.linalg.norm(step_vecs, dim=-1, keepdim=True)

        # Project step_vecs (which hold the transported direction) to the new tangent plane
        step_vecs = step_vecs - torch.sum(step_vecs * new_normals, dim=-1, keepdim=True) * new_normals

        # Restore original magnitude
        new_len = torch.linalg.norm(step_vecs, dim=-1, keepdim=True).clamp(min=1e-8)
        step_vecs = step_vecs * (step_len / new_len)

        # Advance state
        curr_pts = next_pts

        if debug:
            trajectory.append(curr_pts.clone())

    # The accumulated step vectors need to be scaled back up to represent the end direction.
    end_dirs = step_vecs / dt
    end_normals = surface.eval_normal(curr_pts)

    rotation_matrix = None
    if save_parallel_transport:
        end_dirs_unit = end_dirs / torch.linalg.norm(end_dirs, dim=-1, keepdim=True).clamp(min=1e-6)

        start_cross = torch.cross(initial_normals, start_dirs_unit, dim=-1)
        end_cross = torch.cross(end_normals, end_dirs_unit, dim=-1)

        # Start and end orthonormal bases
        start_M = torch.stack([start_dirs_unit, initial_normals, start_cross], dim=-1)
        end_M = torch.stack([end_dirs_unit, end_normals, end_cross], dim=-1)

        # R = (end_M) * (start_M)^T
        rotation_matrix = torch.bmm(end_M, start_M.transpose(-1, -2))

        # Handle zero-length inputs by assigning identity matrix
        eye_batch = torch.eye(3, device=rotation_matrix.device, dtype=rotation_matrix.dtype).unsqueeze(0)
        rotation_matrix = torch.where(idle_idx.unsqueeze(-1).unsqueeze(-1), eye_batch, rotation_matrix)

    debug_tensor = None
    if debug:
        # (B, num_steps + 1, 3)
        debug_tensor = torch.stack(trajectory, dim=1)

    info = ImplicitGeodesicInfo(
        end_directions=end_dirs if save_end_direction else None,
        rotation=rotation_matrix,
        debug_tensor=debug_tensor
    )

    return ImplicitPointBatch(curr_pts), info
