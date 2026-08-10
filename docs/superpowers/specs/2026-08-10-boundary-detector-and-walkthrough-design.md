# Boundary-Complete Detector And Walkthrough Design

## Goal

Implement a CPU-testable assembly of the frozen one-stage nuScenes VoxelNet path and one
executable educational notebook. The detector stops at the explicit sparse-backbone boundary: it
accepts a supplied `SparseBackbone` implementation but does not add a dense fallback or a production
`spconv` backend.

The notebook follows the DiffusionPlanner walkthrough style: it derives the architecture in execution
order, presents every local module as a text I/O card, and executes one synthetic shape trace using
the real package implementation.

## Scope

The detector is responsible for this graph:

```text
VoxelBatch
  -> MeanVoxelFeatureEncoder
  -> SparseBackboneInput(features, [batch,z,y,x], [z,y,x], batch_size)
  -> injected SparseBackbone
  -> RPN
  -> CenterHead
  -> task losses during training
  -> CenterPointPostprocessor during inference
```

It registers only these top-level modules, preserving the frozen state-dict namespace:

```text
reader
backbone
neck
bbox_head
```

The constructor requires explicit instances of the reader, backbone, neck, head, decoder, and
postprocessor dependencies. A small deterministic sparse backend is test-only and is never exposed
as a production default. The detector derives the canonical sparse spatial shape from the frozen
configuration and validates feature/coordinate contracts before invoking the backbone.

Training receives `VoxelBatch` plus six `TaskTargets` and returns the existing per-task loss mapping.
Inference receives `VoxelBatch` and returns `Detections` objects from the existing task-wise
postprocessor. The detector does not own voxelization, dataset loading, augmentation, collation,
production NMS, export, optimizer construction, or checkpoint policy.

## Tests

Detector tests use a small injected `SparseBackbone` that deterministically densifies synthetic
features to the contractually required BEV size. They cover:

- Exact top-level state-dict prefixes and strict checkpoint round trip.
- Mean-VFE to sparse-input coordinate, spatial-shape, batch-size, dtype, and device propagation.
- Empty voxel batches.
- RPN and CenterHead output shapes for all six tasks.
- Training loss keys and backward gradients reaching reader, neck, and head parameters.
- Inference result fields, task offsets, and NMS injection behavior.
- Rejection of incompatible voxel feature dimensions, target task counts, and backbone contracts.

The tests establish local composition only. Official intermediate sparse tensors, a production
`SpMiddleResNetFHD`, and CUDA performance/parity remain separate work.

## Notebook

Create one `notebooks/centerpoint_walkthrough.ipynb`. It imports public package modules and contains
no second implementation. Each major module gets its own markdown/text cell and executable cell:

1. Frozen baseline and an ASCII full-architecture map.
2. Coordinate, box, task, and target conventions.
3. Ten-sweep point loading, augmentation, and class balancing.
4. Hard voxelization and mean VFE.
5. Sparse-backbone interface, with the production backend explicitly marked deferred.
6. Two-stage RPN neck.
7. Shared convolution and six task-specific CenterHead branches.
8. Heatmap/regression losses and target alignment.
9. Dense decoding, rotated-NMS boundary, task offsets, and merge.
10. Full synthetic detector trace.

Every module card specifies purpose, source responsibility, input shape/order/dtype, output
shape/order/dtype, and the contract test that proves it. Executed examples use small tensors except
for a final canonical-dimension shape ledger that does not materialize the canonical sparse grid.
The final trace shows per-stage shapes, six head output shapes, training gradient presence, and
inference detection fields. Notebook outputs remain compact and deterministic.

## Commit Boundaries

1. Add detector contracts and failing assembly tests.
2. Implement the boundary-complete detector and make its focused tests pass.
3. Add complete state/forward/inference tests and documentation updates.
4. Add the executable walkthrough notebook and a notebook execution test.

Each commit runs its focused tests. The final commit sequence runs the complete CPU suite. The
notebook is committed without large outputs or external-data artifacts.

## Explicit Non-Goals

- `spconv` or any production sparse-convolution backend.
- A dense 3D fallback for the canonical grid.
- CUDA rotated-NMS adapter, nuScenes JSON export, devkit evaluation, training engine, DDP, or AMP.
- GT database sampling, tracking, two-stage refinement, double-flip testing, Waymo, PointPillars,
  virtual points, or DCN heads.

## Acceptance Criteria

- The assembled model has the frozen module names and no hidden production fallback.
- Synthetic training and inference run through the real reader, neck, head, decoder, and
  postprocessor with an injected test backbone.
- The full CPU suite passes.
- The notebook executes from a clean kernel and imports package code only.
- Documentation distinguishes boundary-complete assembly from a benchmark-ready detector.
