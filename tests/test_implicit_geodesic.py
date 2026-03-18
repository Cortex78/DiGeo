import torch
import unittest
import numpy as np
from digeo.implicit_surface import ImplicitSurface, ImplicitPointBatch
from digeo.ops.implicit_geodesic import trace_implicit_geodesics


class TestImplicitGeodesic(unittest.TestCase):
    def setUp(self):
        # A simple sphere SDF: f(p) = ||p|| - radius
        self.radius = 2.0

        def sphere_sdf(p: torch.Tensor) -> torch.Tensor:
            return torch.linalg.norm(p, dim=-1, keepdim=True) - self.radius

        self.surface = ImplicitSurface(sdf_func=sphere_sdf)

    def test_geodesic_on_sphere_equator(self):
        """
        Test that a geodesic starting on the equator and moving along it
        stays on the equator and completes a full circle (or arc length).
        """
        # Start at (R, 0, 0)
        starts = ImplicitPointBatch(torch.tensor([[self.radius, 0.0, 0.0]]))

        # Move along the Y axis. Circumference is 2 * pi * R.
        # Let's trace a quarter circle: arc length = pi * R / 2
        arc_length = np.pi * self.radius / 2.0
        dirs = torch.tensor([[0.0, arc_length, 0.0]])

        # Integration parameters
        # dt * max_steps = 1.0 (so total displacement vector magnitude is scaled correctly)
        dt = 0.01
        max_steps = 100

        # Trace
        ends, info = trace_implicit_geodesics(
            self.surface, starts, dirs, dt=dt, max_steps=max_steps, debug=True
        )

        final_pos = ends.positions[0]

        # The analytical endpoint of a quarter circle along the equator from (R, 0, 0)
        # in the +Y direction is (0, R, 0)
        expected_pos = torch.tensor([0.0, self.radius, 0.0])

        # Assert that the final position is close to the expected position
        self.assertTrue(
            torch.allclose(final_pos, expected_pos, atol=1e-2),
            f"Expected {expected_pos}, got {final_pos}"
        )

        # Assert the point is still on the surface (SDF = 0)
        final_sdf = self.surface.eval_sdf(ends.positions)
        self.assertTrue(
            torch.allclose(final_sdf, torch.zeros_like(final_sdf), atol=1e-5),
            f"Point not on surface, SDF value: {final_sdf.item()}"
        )

    def test_geodesic_parallel_transport(self):
        """
        Test that parallel transport works.
        Moving along the equator, a vector pointing initially North (Z)
        should continue pointing North.
        """
        starts = ImplicitPointBatch(torch.tensor([[self.radius, 0.0, 0.0]]))
        arc_length = np.pi * self.radius / 2.0

        # The direction of the geodesic path (East, along Y)
        path_dir = torch.tensor([[0.0, arc_length, 0.0]])

        # Trace and request parallel transport rotation
        ends, info = trace_implicit_geodesics(
            self.surface, starts, path_dir, dt=0.01, max_steps=100, save_parallel_transport=True
        )

        # We want to transport a vector that initially points North (0, 0, 1)
        v_in = torch.tensor([[0.0, 0.0, 1.0]])

        # Apply the transported rotation
        v_out = info.transport(v_in)[0]

        # On a sphere, transporting a purely orthogonal vector along a great circle
        # should leave it pointing in the same global orientation (0, 0, 1)
        expected_v_out = torch.tensor([0.0, 0.0, 1.0])
        self.assertTrue(
            torch.allclose(v_out, expected_v_out, atol=1e-2),
            f"Expected {expected_v_out}, got {v_out}"
        )

if __name__ == "__main__":
    unittest.main()
