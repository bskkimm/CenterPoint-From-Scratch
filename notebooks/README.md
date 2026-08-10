# Educational Notebooks

This directory contains executable architecture walkthroughs. Notebooks import the package
implementation rather than maintain a second model implementation, and are committed without
large generated outputs.

`centerpoint_walkthrough.ipynb` is a boundary-complete VoxelNet walkthrough. It injects a tiny
notebook-only sparse backend for its small execution example because `SpMiddleResNetFHD` remains
an injected, unimplemented production backend.

```bash
jupyter notebook notebooks/centerpoint_walkthrough.ipynb
```
