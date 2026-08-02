# Tests

This directory contains deterministic tests for coordinate and tensor contracts, voxelization,
targets, losses, decoding, data preprocessing, model modules, checkpoint state, and correctness
oracles. Official-reference values are stored in `fixtures/`; `reference/` contains the pinned
fixture-generation workflow.

Run the CPU suite with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q
```
