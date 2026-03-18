import torch
from torch import Tensor
from typing import Callable, Optional, Union, Tuple, overload
from digeo.mesh import get_device_dtype


class ImplicitSurface:
    def __init__(
        self,
        sdf_func: Callable[[Tensor], Tensor],
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize an ImplicitSurface object.

        Parameters
        ----------
        sdf_func : Callable[[Tensor], Tensor]
            A function or neural network module that takes a batched tensor of 3D points
            (B, 3) and returns their signed distance (B, 1).
        device : str | torch.device
            The device on which computations should occur.
        dtype : torch.dtype
            The data type to use.
        """
        self.sdf_func = sdf_func
        self.device = device
        self.dtype = dtype

    def eval_sdf(self, points: Tensor) -> Tensor:
        """
        Evaluate the signed distance at the given points.
        """
        return self.sdf_func(points)

    def eval_normal(self, points: Tensor) -> Tensor:
        """
        Evaluate the normal (gradient of the SDF) at the given points.
        The gradient points outwards from the surface.
        """
        with torch.enable_grad():
            points_req_grad = points.requires_grad_(True)
            sdf_val = self.sdf_func(points_req_grad)

            grad_outputs = torch.ones_like(sdf_val)
            grad_sdf = torch.autograd.grad(
                outputs=sdf_val,
                inputs=points_req_grad,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

        return torch.nn.functional.normalize(grad_sdf, p=2, dim=-1)

    @overload
    def to(self, device: Union[str, torch.device]) -> "ImplicitSurface": ...

    @overload
    def to(self, dtype: torch.dtype) -> "ImplicitSurface": ...

    @overload
    def to(self, device: Union[str, torch.device], dtype: torch.dtype) -> "ImplicitSurface": ...

    @overload
    def to(self, tensor: "Tensor") -> "ImplicitSurface": ...

    @overload
    def to(
        self,
        *,
        device: Optional[Union[str, torch.device]] = ...,
        dtype: Optional[torch.dtype] = ...,
    ) -> "ImplicitSurface": ...

    def to(self, *args, **kwargs) -> "ImplicitSurface":
        """
        Move the surface evaluation to a different device and/or dtype.
        (Note: For complex neural networks, the `sdf_func` must handle its own `to()` calls
        if it holds internal parameters. This simple update just changes the wrapper's config).
        """
        device, dtype = get_device_dtype(*args, **kwargs)

        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype

        if hasattr(self.sdf_func, 'to'):
            self.sdf_func = self.sdf_func.to(device=self.device, dtype=self.dtype)

        return self


class ImplicitPointBatch:
    def __init__(self, positions: Tensor):
        """
        Initialize a batch of implicit surface points.

        Unlike a MeshPointBatch which relies on (face, uv) pairs, an
        ImplicitPointBatch directly stores the 3D coordinates.

        Parameters
        ----------
        positions : Tensor
            (B, 3) tensor of point coordinates.
        """
        if positions.dim() != 2 or positions.size(1) != 3:
            raise ValueError("positions must be a 2D tensor of shape (B, 3).")

        self.positions: Tensor = positions
        self.positions.requires_grad_()

    @overload
    def to(self, device: Union[str, torch.device]) -> "ImplicitPointBatch": ...

    @overload
    def to(self, dtype: torch.dtype) -> "ImplicitPointBatch": ...

    @overload
    def to(
        self, device: Union[str, torch.device], dtype: torch.dtype
    ) -> "ImplicitPointBatch": ...

    @overload
    def to(self, tensor: "Tensor") -> "ImplicitPointBatch": ...

    @overload
    def to(
        self,
        *,
        device: Optional[Union[str, torch.device]] = ...,
        dtype: Optional[torch.dtype] = ...,
    ) -> "ImplicitPointBatch": ...

    def to(self, *args, **kwargs) -> "ImplicitPointBatch":
        """
        Move the points to a different device and/or dtype.
        """
        device, dtype = get_device_dtype(*args, **kwargs)

        if device is None:
            device = str(self.positions.device)
        if dtype is None:
            dtype = self.positions.dtype

        return ImplicitPointBatch(
            positions=self.positions.to(device=device, dtype=dtype)
        )

    def detach(self) -> "ImplicitPointBatch":
        return ImplicitPointBatch(positions=self.positions.detach())

    def clone(self) -> "ImplicitPointBatch":
        return ImplicitPointBatch(positions=self.positions.clone())

    def interpolate(self, return_batch: bool = False) -> Tensor:
        """
        For implicit surfaces, the coordinates are already explicit.
        This method is primarily to match the API of `MeshPointBatch`.
        """
        if return_batch:
            return self.positions.unsqueeze(0)
        return self.positions

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: Union[int, slice, Tensor]) -> "ImplicitPointBatch":
        if isinstance(idx, int):
            return ImplicitPointBatch(positions=self.positions[idx].unsqueeze(0))
        elif isinstance(idx, (slice, Tensor)):
            return ImplicitPointBatch(positions=self.positions[idx])
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")
