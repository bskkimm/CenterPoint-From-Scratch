# Model State-Dict Contract

Model-state schema version **1** fixes module namespaces before trainable backbone and head modules
are implemented. Checkpoint container version and model-state version are separate because training
metadata can evolve without renaming model parameters.

## Canonical Top-Level Names

The detector must register these modules directly:

```text
reader
backbone
neck
bbox_head
```

These names follow the pinned official detector and avoid translation for the top-level checkpoint
structure. Do not rename `bbox_head` to `head`, and do not insert framework registries or generic
`model` wrappers into parameter keys.

## Canonical Submodules

The sparse backbone uses:

```text
backbone.conv_input
backbone.conv1
backbone.conv2
backbone.conv3
backbone.conv4
backbone.extra_conv
```

Residual blocks use `conv1`, `bn1`, `conv2`, `bn2`, and `downsample`. Backend-specific sparse tensor
objects are not modules and must not add state-dict levels.

The BEV neck uses:

```text
neck.blocks.{stage_index}
neck.deblocks.{stage_index}
```

The detection head uses:

```text
bbox_head.shared_conv
bbox_head.tasks.{task_index}.{branch}
```

Task indices follow the six-task order in `docs/CONTRACTS.md`. Branch names are exactly `reg`,
`height`, `dim`, `rot`, `vel`, and `hm`. Numeric sequential-layer indices follow construction order
and cannot be changed after checkpoint parity is established.

## Wrapper And Backend Rules

- Save the unwrapped detector state; canonical keys never begin with `module.`.
- DDP, compilation, profiling, and precision wrappers must not alter persisted keys.
- A sparse backend must not insert names such as `backend`, `implementation`, or `impl` below the
  canonical backbone stages.
- Official or third-party checkpoints with different keys are translated by a dedicated importer.
  Translation logic does not add aliases to the model itself.
- A backend update that changes parameter layout requires a reviewed importer or a new model-state
  schema version.

## Validation Gates

Before each trainable module merges, tests must snapshot and review:

1. Exact state-dict keys and tensor shapes
2. Trainable versus non-trainable entries
3. Initialization values or statistics
4. Save/load equality through the versioned checkpoint container
5. Mapping coverage for the pinned official checkpoint

The final detector merge must include a checked fixture of all keys and shapes. Updating that
fixture requires an explicit model-state schema review; it is not a routine test refresh.
