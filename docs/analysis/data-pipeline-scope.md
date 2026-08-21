# Data Pipeline Scope Decisions

## Ground-Truth Database Sampling

The pinned configuration supplies a database-sampler mapping whose `enable` field is `False`.
However, the pinned preprocessing constructor checks only whether that mapping is `None`, and its
sampler builder does not consume `enable`. Literal execution can therefore sample database objects
despite the configuration's disabled flag.

The frozen local baseline follows the declared `enable=False` intent and does not implement or run
ground-truth database sampling. Adding it would require a new reviewed baseline decision and parity
fixtures rather than an implicit change to training data.

## Randomness And Mutation

Augmentation and sweep loading accept an injectable NumPy-compatible RNG while defaulting to
`numpy.random`. The default draw order and float32 assignment behavior match the pinned helpers;
injection exists so tests can prove that order without changing global process state.

Local augmentation returns transformed copies instead of mutating caller-owned arrays. This does
not change produced points or boxes, but prevents sample-cache aliasing from becoming part of the
public data interface.

## Integrated Dataset Path

`CenterPointDataset` composes the tested stages in the pinned order and gates augmentation and BEV
range filtering to training mode, matching the pinned preprocessing and voxelization stages. Two
deliberate differences remain.

First, target assignment is not gated to training mode. Upstream assigns labels only when the
pipeline runs in training mode, so its validation batches carry no targets. The merged collation
contract requires a uniform, non-zero task count across a batch, so gating assignment would make
validation batches uncollatable. Assignment therefore runs in both modes and `assign_targets=False`
is available for callers that want the upstream shape; such samples cannot be collated. Revisit this
when a validation loop exists and the intended validation contract is settled.

Second, metadata records are injected instead of built. The nuScenes metadata builder is gated on
official data, so the dataset consumes records providing `REQUIRED_INFO_KEYS`. This fixes the seam
the future builder must emit, and the fixed-token parity gate stays open until it exists.

The reference voxelizer iterates points in Python. The integrated path is therefore sized for
contract tests and small samples; a production voxelization backend is still required before this
path can feed full-speed training.
