# Architectural Analysis: Integrating Signed Distance Functions (SDFs) into DiGeo

## 1. Overview and Current Architecture

DiGeo currently operates strictly on explicit triangle meshes. The primary geometric operations (geodesics, parallel transport, and exponential map) are formulated around discrete mesh primitives:
* **Vertices and Faces:** Define the geometry.
* **Adjacencies:** Determine how triangles are connected (used for "triangle hopping" when tracing geodesics).
* **Barycentric Coordinates / UVs:** Define exact positions on the mesh surface.

The core computational workhorse is the `straightest_geodesic` function (and its CUDA counterpart). This function takes a starting face, local UV coordinates, and a 3D direction vector. It traces the ray across triangles, calculating edge/vertex crossings, and applies parallel transport at these discrete boundaries.

## 2. The SDF Paradigm

A Signed Distance Function (SDF) implicitly defines a surface as the zero level-set of a continuous function $f(x, y, z)$.
* **Surface Points:** Points where $f(p) = 0$.
* **Normals:** The gradient of the SDF, $\nabla f(p)$, normalized.
* **Geodesics:** Instead of triangle hopping, tracing a geodesic on an SDF requires numerically integrating an Ordinary Differential Equation (ODE).

### Geodesic Equation on an Implicit Surface

To trace a straightest geodesic (or exponential map) on an implicit surface, we start at point $p_0$ with an initial tangent vector $v_0$ (where $v_0 \cdot \nabla f(p_0) = 0$).

A standard numerical approach (e.g., using forward Euler or Runge-Kutta) involves:
1. **Stepping in the tangent direction:** $p' = p_i + \Delta t \cdot v_i$
2. **Projecting back to the surface:** Since the step moves off the curved surface, we must project $p'$ back to the zero level-set. This is typically done using Newton-Raphson steps along the gradient: $p_{i+1} = p' - f(p') \frac{\nabla f(p')}{|\nabla f(p')|^2}$
3. **Parallel Transporting the direction:** The direction vector $v$ must be updated to remain tangent to the new point $p_{i+1}$. This is done by projecting $v_i$ onto the new tangent plane and normalizing: $v_{i+1} = v_i - (v_i \cdot n_{i+1}) n_{i+1}$, where $n_{i+1} = \frac{\nabla f(p_{i+1})}{|\nabla f(p_{i+1})|}$.

## 3. Required API Abstractions

To support SDFs without breaking the existing mesh-based API, we need parallel abstractions:

| Mesh Concept | Implicit Concept | Description |
| :--- | :--- | :--- |
| `Mesh` | `ImplicitSurface` | Represents the surface. For SDFs, it wraps a callable or PyTorch module that computes $f(p)$. |
| `MeshPointBatch` | `ImplicitPointBatch` | Represents points on the surface. Instead of `(face, uv)`, it stores explicit 3D coordinates `(x, y, z)`. |
| `trace_geodesics` (Mesh) | `trace_implicit_geodesics` | Traces the path using ODE integration and projection, rather than discrete triangle hopping. |

## 4. Implementation Challenges and Strategy

1. **Gradients of the SDF:**
   To project points and compute normals, we need $\nabla f(p)$. If the SDF is a neural network (e.g., in PyTorch), we can use `torch.autograd.grad`. However, computing double gradients (if the user wants to backpropagate through the geodesic tracing itself) can become computationally expensive.
2. **Batching:**
   The ODE solver must handle batched points. Some points may converge to the surface faster than others during the projection step. We will need a fixed number of projection steps or masked updates to keep tensor operations batched and efficient.
3. **Integration with CUDA:**
   Currently, DiGeo uses custom C++/CUDA kernels for mesh geodesics. For SDFs (especially neural SDFs), the evaluation of $f(p)$ happens in standard PyTorch. Therefore, the prototype for implicit geodesics will be implemented purely in PyTorch using standard tensor operations. This allows automatic differentiation and leverages PyTorch's native GPU acceleration.
4. **Step Size ($\Delta t$):**
   Unlike mesh geodesics which take exact geometric steps to triangle boundaries, ODE integration requires a carefully chosen step size. Too large, and the projection fails; too small, and it is inefficient.

## 5. Prototype Plan

The prototype will introduce:
1. `src/digeo/implicit_surface.py`: Defining `ImplicitSurface` and `ImplicitPointBatch`.
2. `src/digeo/ops/implicit_geodesic.py`: Implementing the ODE-based geodesic tracing with surface projection.
3. `tests/test_implicit_geodesic.py`: A test script demonstrating tracing a geodesic around a perfect sphere (defined analytically via SDF: $f(p) = |p| - r$).
