# Repository Instructions For Coding Agents

## Mission

Build a faithful, educational reimplementation of the pinned one-stage CenterPoint nuScenes
VoxelNet. Correctness and reproducible evidence take priority over feature count or speed.

Read these files before changing code:

1. `docs/BASELINE.md` defines the exact reproduction target.
2. `docs/MATERIALS.md` lists the primary sources.
3. `docs/NEXT_STEPS.md` defines the implementation order and merge gates.

## Source Of Truth

Use sources in this order when they disagree:

1. The official repository commit and configuration pinned in `docs/BASELINE.md`
2. Official nuScenes data and evaluation contracts
3. The CenterPoint paper for method intent
4. Secondary explanations only for orientation

Do not silently combine the paper-era 0.10 m model with the pinned 0.075 m repository model.
Do not add tracking, Waymo, DCN, double-flip inference, PointPillars, virtual points, or two-stage
refinement to the initial reproduction path.

## Main Branch Standard

Code is suitable for `main` only when its behavior is supported by proportionate evidence:

- Mathematical primitives require deterministic unit tests and official-reference parity where
  equivalent source can be executed.
- Data and coordinate code requires fixed-token or fixed-array parity, boundary cases, and
  round-trip checks.
- Model modules require shape, initialization, state-dict, forward, and gradient tests.
- Backend adapters require parity against a slow reference and explicit version constraints.
- Training changes require a one-batch overfit or short controlled run before full experiments.
- Accuracy claims require the exact config, environment, checkpoint, logs, prediction file, and
  official devkit output.

Keep exploratory backends, hyperparameters, optimizations, and ablations on feature or experiment
branches until their evidence gate is satisfied. Never describe an official model-zoo number as a
result produced by this repository.

## Implementation Rules

- Preserve the internal box layout `[x, y, z, w, l, h, vx, vy, yaw]`.
- Preserve target order `[dx, dy, z, log(w), log(l), log(h), vx, vy, sin(yaw), cos(yaw)]`.
- Preserve hard-voxel first-occurrence order, `zyx` coordinates, and point/voxel truncation rules.
- Preserve six-task nuScenes class grouping and task-local labels.
- Preserve dense decoding without adding top-K or local-maximum filtering.
- Keep sparse convolution, production voxelization, and rotated NMS behind explicit interfaces.
- Use official dataset devkits for benchmark metrics instead of reimplementing them.
- Prefer small modules with explicit tensor shape and coordinate contracts.
- Do not add compatibility behavior without a concrete persisted or external requirement.

## Verification

Run the CPU suite before every proposed merge:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q
```

Also run focused tests for changed modules. Regenerate official fixtures only from a clean checkout
at the pinned commit:

```bash
python3 tests/reference/generate_official_fixtures.py /path/to/official/CenterPoint \
  > /tmp/official_3cf7d870.json
```

Compare regenerated output with `tests/fixtures/official_3cf7d870.json`; do not replace a fixture
without explaining and reviewing every difference.

## Documentation And Commits

- Record deliberate differences from upstream in `docs/analysis/`.
- Document commands only after verifying that they run in the declared environment.
- Keep generated datasets, checkpoints, logs, and large predictions out of Git.
- Keep commits focused by contract: implementation, tests, fixture updates, and experiment evidence
  may be separate commits when independently reviewable.
- Do not mark roadmap items complete when only a primitive or placeholder exists.
