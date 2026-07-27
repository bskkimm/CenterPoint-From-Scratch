# Backend Boundaries

## Decision

Core model semantics remain local PyTorch code. Operations that require specialized kernels or
official benchmark implementations are isolated behind explicit interfaces.

| Boundary | Correctness reference | Intended production path |
| --- | --- | --- |
| Hard voxelization | Ordered CPU implementation in this repository | Reference CPU path first; optimized adapter only after exact parity |
| Sparse 3D convolution | Tiny dense cases and pinned official intermediate tensors | Version-pinned `spconv` adapter |
| Rotated IoU and NMS | Slow local geometric oracle on small inputs | Version-pinned compiled operator adapter |
| Detection metrics | Official prediction fixture | Official nuScenes devkit |

## Rationale

The canonical sparse grid is too large for a faithful dense `Conv3d` replacement. Reimplementing a
production sparse-convolution library would increase risk without improving understanding of
CenterPoint. Likewise, replacing rotated NMS with circle or axis-aligned NMS would change the
frozen model rather than remove a dependency.

“From scratch” therefore means that CenterPoint data semantics, model assembly, heads, targets,
losses, decoding, training, and experiment control are implemented locally. It does not mean
reimplementing general-purpose sparse kernels or official benchmark metrics.

## Adapter Requirements

- Core code must not import a specific backend outside its adapter module.
- Inputs, outputs, coordinate order, dtype, device, and determinism behavior must be documented.
- Every optimized backend must match a slow reference on boundary, empty, tie, and randomized cases.
- Backend and framework versions must be recorded in checkpoints and experiment manifests.
- Backend changes require parity and performance evidence; they are not dependency-only updates.
- Missing optional backends should produce an actionable error, not silently select different model
  behavior.

## Deferred Decisions

Exact PyTorch, CUDA, `spconv`, and rotated-NMS package versions remain open until installation,
checkpoint loading, intermediate parity, and target hardware have been tested together. Those
versions belong in the environment lock, not in this architecture decision.
