# Evidence-Gated Next Steps

This plan advances the frozen nuScenes one-stage VoxelNet target in `BASELINE.md`. It distinguishes
code that can be proven with deterministic parity from work that requires real data, GPUs, or full
training.

## Current Foundation

The repository already contains tested implementations of:

- Internal box conventions and 3D corners
- Gaussian heatmaps and indexed tensor gathering
- Heatmap focal loss and gathered regression loss
- Ordered hard voxelization and mean voxel feature encoding
- Six-task target assignment
- Dense CenterHead decoding before NMS
- Golden fixtures generated from the pinned official implementation

These are primitives, not yet a complete detector or reproduction.

## 1. Lock The Environment And Configuration

**Deliverables**

- Declare supported Python, PyTorch, CUDA, cuDNN, `spconv`, nuScenes devkit, and NumPy versions.
- Add a machine-readable canonical nuScenes configuration without importing the official framework.
- Validate all frozen values against the pinned config blob.
- Record dependency installation and environment inspection commands.

**Merge gate**

- Clean-environment installation succeeds.
- A config parity test checks every architecture, geometry, loss, training, and inference field used
  by this repository.

## 2. Implement nuScenes Metadata And Sweep Loading

**Deliverables**

- Dataset split and category mapping
- Ten-sweep selection and time-lag feature
- Sensor-to-reference-LiDAR transformations
- Box, velocity, and yaw conversion into the internal convention
- Metadata cache with an explicit schema and version

**Merge gate**

- Fixed sample-token outputs match the pinned official preprocessing.
- Tests cover missing history, near-origin sweep filtering, transforms, timestamps, and empty boxes.
- Dataset counts, class histograms, and sweep distributions are recorded for `v1.0-trainval`.

## 3. Implement Training Preprocessing And Batching

**Deliverables**

- Point shuffling and class-balanced group sampling
- Global rotation, scale, translation, and flip augmentation
- Optional ground-truth database interface, disabled for the frozen baseline
- Batch collation with prepended sparse batch coordinates

**Merge gate**

- Seeded transformation fixtures match official behavior.
- Box/point invariants and empty-sample behavior are tested.
- Batch tensor shapes and task target lists are covered for batch sizes one and greater than one.

## 4. Establish The Sparse-Convolution Boundary

The canonical `1440 x 1440 x 40` grid makes a dense 3D replacement impractical. The production
path should use an explicit `spconv` adapter while retaining tiny reference cases for correctness.

**Deliverables**

- Sparse tensor adapter with documented coordinate order
- `SpMiddleResNetFHD` stages and residual blocks
- Dense BEV output contract `[B, 256, 180, 180]`
- Versioned `spconv` compatibility policy

**Merge gate**

- Stage shapes, state-dict layout, sparse coordinates, and fixed-input outputs match upstream.
- Tiny dense/reference convolution cases verify active-site semantics.
- CUDA memory and forward latency are measured, but not yet presented as paper-speed reproduction.

## 5. Implement The BEV Neck And CenterHead

**Deliverables**

- Official two-block BEV RPN neck
- Shared task feature convolution
- Six task-specific heatmap and regression heads
- Official initialization, head order, loss composition, and code weights

**Merge gate**

- Parameter counts, state-dict keys, output shapes, initialization statistics, forward values, and
  gradients match a fixed official checkpoint or extracted reference modules.
- Empty-target and multi-task loss integration tests pass.

## 6. Assemble The Detector

**Deliverables**

- `voxelize -> mean VFE -> sparse backbone -> BEV neck -> CenterHead`
- Training and inference result contracts
- Device transfer and mixed-precision boundaries
- Checkpoint save/load with complete reproducibility state

**Merge gate**

- Synthetic end-to-end forward and backward pass succeeds.
- One real nuScenes sample matches intermediate official tensor shapes and selected values.
- A tiny sample can overfit without NaNs or disconnected gradients.

## 7. Implement Canonical Postprocessing And Evaluation Export

**Deliverables**

- Slow rotated IoU/NMS correctness oracle
- Production rotated-NMS adapter
- Task-wise NMS and global label merge
- LiDAR-to-global box and velocity conversion
- nuScenes attributes and result JSON validation
- Official devkit evaluation wrapper

**Merge gate**

- Pre/post-NMS outputs match the official checkpoint on fixed samples.
- Golden tests cover ties, empty candidates, thresholds, range boundaries, and all task offsets.
- The official devkit accepts a complete validation prediction file without repair.

## 8. Implement The Training Engine

**Deliverables**

- Adam with the pinned fixed-weight-decay behavior
- One-cycle schedule, gradient clipping, DDP, SyncBN, seeding, and resume
- Structured logs, configuration snapshot, environment manifest, and checkpoint hashes

**Merge gate**

- Learning-rate trace and optimizer updates match the intended recipe.
- Resume reproduces uninterrupted training state.
- Single-process and multi-process smoke tests pass.
- One-batch and small-subset overfit runs are archived before full training.

## 9. Run Reproduction Experiments

Perform this work on experiment branches. Do not tune the frozen baseline while calling it a
reproduction.

**Required artifacts**

- Git commit and canonical config hash
- Environment and hardware manifest
- Dataset metadata hashes
- Seed, effective batch size, and complete launch command
- Checkpoints, training curves, logs, predictions, and official metric summaries
- Per-class AP and TP errors, not only aggregate mAP/NDS

The first benchmark claim should target the pinned **59.6 mAP / 66.8 NDS** validation row. Define
the acceptance tolerance before training, and investigate class-level regressions even when the
aggregate target is met.

## 10. Publish The Educational Walkthrough

Build notebooks from the tested package rather than creating a second implementation. Start after
the corresponding modules are stable.

Recommended order:

1. Coordinates, sweeps, and voxelization
2. Sparse backbone to BEV
3. Heatmaps and target assignment
4. Losses and gradients
5. Dense decoding, NMS, and nuScenes export
6. End-to-end model and reproduction artifacts

Every notebook should show tensor shapes, small numerical examples, failure cases, and links to the
tests that establish the behavior.

## Immediate Work Queue

1. Environment lock and canonical local config
2. nuScenes metadata/sweep loader with fixed-token parity
3. Augmentation and collation
4. Sparse backend adapter and `SpMiddleResNetFHD`
5. BEV neck and CenterHead

Items 1-3 can progress mostly on CPU. Items 4 onward require a declared CUDA and `spconv`
environment; full reproduction claims remain blocked on end-to-end training and official evaluation.
