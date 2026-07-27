# Contributing

Contributions should preserve the frozen target in `docs/BASELINE.md` and follow the implementation
sequence in `docs/NEXT_STEPS.md`.

## Before Opening A Pull Request

1. Identify the pinned official source or dataset contract governing the behavior.
2. Keep the change focused on one explicit tensor, coordinate, module, or experiment contract.
3. Add deterministic tests proportional to the risk.
4. Record deliberate differences from upstream in `docs/analysis/`.
5. Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q
```

## Evidence Expectations

- Geometry and data changes need boundary, round-trip, and official fixture/token parity tests.
- Model changes need initialization, shape, state-dict, forward, and gradient tests.
- Backend changes need slow-reference parity and version documentation.
- Training changes need a controlled smoke or overfit run before a full experiment.
- Result claims need complete configuration, environment, checkpoint, log, prediction, and official
  evaluation artifacts.

Do not update official golden fixtures merely to make a test pass. Regenerate them from a clean
checkout at the pinned commit and review every difference.

## Scope

Tracking, Waymo, optional refinement, DCN, PointPillars, virtual points, double-flip inference, and
paper-era geometry are not part of the first reproduction. Develop them separately and do not add
compatibility branches to the canonical path without an approved target change.

Generated datasets, checkpoints, logs, experiment stores, and prediction files must remain outside
Git history.
