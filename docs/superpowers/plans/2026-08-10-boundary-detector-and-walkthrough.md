# Boundary Detector And Walkthrough Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Assemble the frozen VoxelNet stages around an injected sparse backbone and publish one executable module-by-module educational walkthrough notebook.

**Architecture:** `VoxelNet` owns the canonical `reader`, `backbone`, `neck`, and `bbox_head` module namespaces. It converts `VoxelBatch` into `SparseBackboneInput`, then delegates training losses to `CenterHead` and inference to `CenterPointPostprocessor`. The notebook imports those public modules, explains each contract in execution order, and uses a small test-only backbone to execute a synthetic trace.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, pytest, Jupyter notebook JSON (`nbformat` only if already installed).

## Global Constraints

- Preserve the pinned `3cf7d870537e287c99b43b68636ea392a5e6f519` nuScenes 0.075 m target.
- Register detector modules directly as `reader`, `backbone`, `neck`, and `bbox_head`; do not add a wrapper namespace.
- Preserve `[batch,z,y,x]` sparse coordinates and `[x,y,z,w,l,h,vx,vy,yaw]` internal boxes.
- Production sparse convolution remains behind `SparseBackbone`; do not add a dense fallback or default fake backend.
- Keep task order, head branch order, dense decoding, task-wise NMS, and global-label merge unchanged.
- The walkthrough imports package code and never creates a second detector implementation.
- No CUDA, `spconv`, nuScenes data, notebook-generated datasets, or large notebook outputs.
- Run `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q` before the final commit.

---

## File Structure

- `centerpoint/models/detectors/voxelnet.py`: detector assembly and typed training/inference entry points.
- `centerpoint/models/detectors/__init__.py`: public detector exports.
- `centerpoint/models/__init__.py`: public `VoxelNet` export.
- `tests/test_voxelnet.py`: test-only sparse backend and detector composition tests.
- `tests/test_notebook_walkthrough.py`: static/execution validation for the walkthrough.
- `notebooks/centerpoint_walkthrough.ipynb`: one module-by-module executable educational walkthrough.
- `notebooks/README.md`: notebook command and scope update.
- `docs/UPSTREAM_MAP.md` and `docs/NEXT_STEPS.md`: boundary-complete detector status, without claiming `spconv` parity.

### Task 1: Specify Detector API With Failing Tests

**Files:**
- Create: `tests/test_voxelnet.py`
- Modify: `centerpoint/models/__init__.py`

**Interfaces:**
- Consumes: `VoxelBatch`, `TaskTargets`, `MeanVoxelFeatureEncoder`, `SparseBackbone`, `RPN`, `CenterHead`, `CenterPointPostprocessor`, and `NUSCENES_VOXELNET_075`.
- Produces: required public API `VoxelNet`, `VoxelNet.forward_features(voxels: VoxelBatch) -> tuple[list[dict[str, Tensor]], Tensor]`, `VoxelNet.loss(voxels: VoxelBatch, targets: Sequence[TaskTargets]) -> dict[str, list[Tensor]]`, and `VoxelNet.predict(voxels: VoxelBatch) -> list[Detections]`.

- [ ] **Step 1: Write the failing detector construction and forward test**

```python
def test_voxelnet_registers_frozen_top_level_modules_and_traces_features():
    model = make_model()
    predictions, bev = model.forward_features(make_voxel_batch())

    assert tuple(model._modules) == ("reader", "backbone", "neck", "bbox_head")
    assert bev.shape == (2, 512, 4, 4)
    assert [task["hm"].shape[1] for task in predictions] == [1, 2, 2, 1, 2, 2]
```

Define `TestSparseBackbone(SparseBackbone)` in this test file. Its `forward_sparse()` must create a dense `[B,256,H,W]` output from `inputs.features.sum(dim=1)` without parameters and retain empty-input support. Use an 8x downsampled synthetic spatial shape `(4, 32, 32)` so RPN produces `[B,512,4,4]`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/test_voxelnet.py::test_voxelnet_registers_frozen_top_level_modules_and_traces_features`

Expected: collection fails because `VoxelNet` cannot be imported from `centerpoint.models`.

- [ ] **Step 3: Write failing training, inference, and validation tests**

```python
def test_voxelnet_training_loss_backpropagates_through_trainable_stages():
    losses = make_model().loss(make_voxel_batch(), make_task_targets(batch_size=2))
    total = sum(losses["loss"])
    total.backward()
    assert model.neck.blocks[0][1].weight.grad is not None
    assert model.bbox_head.shared_conv[0].weight.grad is not None

def test_voxelnet_predict_merges_six_tasks():
    detections = make_model().eval().predict(make_voxel_batch())
    assert len(detections) == 2
    assert detections[0].boxes.shape[1] == 9

def test_voxelnet_rejects_incompatible_targets_and_feature_channels():
    with pytest.raises(ValueError, match="task count"):
        make_model().loss(make_voxel_batch(), make_task_targets()[:5])
```

Make target heatmaps `[B,C,4,4]`, use object index `0`, one valid center per task, and retain all target fields required by `TaskTargets`. Add an empty-voxel-batch test with `[0,10,5]` voxels, `[0]` counts, and `[0,4]` coordinates.

- [ ] **Step 4: Run the full focused file to verify all tests fail for missing implementation**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/test_voxelnet.py`

Expected: collection fails only because `VoxelNet` is unavailable; no test helpers should error independently.

- [ ] **Step 5: Commit the test contract**

```bash
git add tests/test_voxelnet.py
git commit -m "test: specify VoxelNet assembly contract"
```

### Task 2: Implement Boundary-Complete VoxelNet

**Files:**
- Create: `centerpoint/models/detectors/__init__.py`
- Create: `centerpoint/models/detectors/voxelnet.py`
- Modify: `centerpoint/models/__init__.py`
- Test: `tests/test_voxelnet.py`

**Interfaces:**
- Consumes: the API specified in Task 1.
- Produces: importable `VoxelNet` that contains only `reader`, `backbone`, `neck`, and `bbox_head` as direct registered module names.

- [ ] **Step 1: Implement the constructor and canonical sparse input conversion**

```python
class VoxelNet(nn.Module):
    def __init__(self, reader, backbone, neck, bbox_head, postprocessor, spatial_shape):
        super().__init__()
        self.reader = reader
        self.backbone = backbone
        self.neck = neck
        self.bbox_head = bbox_head
        self._postprocessor = postprocessor
        self._spatial_shape = tuple(spatial_shape)

    def _sparse_inputs(self, voxels: VoxelBatch) -> SparseBackboneInput:
        features = self.reader(voxels.voxels, voxels.num_points)
        return SparseBackboneInput(features, voxels.coordinates, self._spatial_shape, voxels.batch_size)
```

Validate that `spatial_shape` has three positive values and that VFE output feature channels match the backbone’s expected interface. Keep `_postprocessor` out of the state dict by storing it as a non-module callable only if necessary; otherwise accept it only in `predict()` to avoid violating the frozen four top-level module contract.

- [ ] **Step 2: Implement feature, loss, and inference methods**

```python
def forward_features(self, voxels: VoxelBatch) -> tuple[list[dict[str, Tensor]], Tensor]:
    sparse = self._sparse_inputs(voxels)
    bev = self.backbone(sparse)
    neck_features = self.neck(bev)
    return self.bbox_head(neck_features)

def loss(self, voxels: VoxelBatch, targets: Sequence[TaskTargets]) -> dict[str, list[Tensor]]:
    predictions, _ = self.forward_features(voxels)
    return self.bbox_head.loss(predictions, targets)

@torch.no_grad()
def predict(self, voxels: VoxelBatch) -> list[Detections]:
    predictions, _ = self.forward_features(voxels)
    return self.postprocessor(predictions)
```

Use the actual `CenterHead.loss()` and `CenterPointPostprocessor`. Reject an empty target sequence before entering `CenterHead.loss()` with an error containing `task count`. Derive `spatial_shape` in a config factory from `config.voxel.grid_size` reordered to `[z,y,x]`, i.e. `(grid_z, grid_y, grid_x)`.

- [ ] **Step 3: Export and build from the frozen configuration**

Add `VoxelNet` to `centerpoint/models/detectors/__init__.py` and `centerpoint/models/__init__.py`. Add a narrow `CenterPointConfig.make_voxelnet(backbone, postprocessor)` factory that creates `MeanVoxelFeatureEncoder`, `RPN`, and `CenterHead` using only existing frozen config fields, and requires the caller to supply the backbone and postprocessor.

- [ ] **Step 4: Run focused tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/test_voxelnet.py tests/test_rpn_neck.py tests/test_center_head.py tests/test_postprocess.py`

Expected: PASS.

- [ ] **Step 5: Commit the detector assembly**

```bash
git add centerpoint/config.py centerpoint/models/__init__.py centerpoint/models/detectors tests/test_voxelnet.py
git commit -m "feat: assemble boundary-complete VoxelNet"
```

### Task 3: Lock Assembly State And Composition Evidence

**Files:**
- Modify: `tests/test_voxelnet.py`
- Modify: `tests/test_model_state.py`
- Modify: `docs/UPSTREAM_MAP.md`
- Modify: `docs/NEXT_STEPS.md`

**Interfaces:**
- Consumes: `VoxelNet` from Task 2 and versioned `save_checkpoint()` / `load_checkpoint()`.
- Produces: explicit state-dict and behavior evidence for boundary-complete assembly, without claiming a production sparse backend.

- [ ] **Step 1: Write failing full-prefix and checkpoint tests**

```python
def test_voxelnet_state_uses_only_frozen_top_level_prefixes():
    state = make_model().state_dict()
    assert {name.split(".")[0] for name in state} == {"neck", "bbox_head"}
    assert all(not name.startswith("postprocessor.") for name in state)

def test_voxelnet_checkpoint_round_trip_preserves_composed_outputs(tmp_path):
    model = make_model().eval()
    expected, _ = model.forward_features(make_voxel_batch())
    save_checkpoint(path, model=model, config={}, epoch=0, global_step=0)
    load_checkpoint(path, model=model)
    actual, _ = model.forward_features(make_voxel_batch())
    assert_task_maps_equal(actual, expected)
```

The expected top-level prefixes omit parameter-free `reader` and test-only parameter-free `backbone`; this proves they remain module namespaces without inventing buffers. Seed the model before construction and compare every branch map with `torch.testing.assert_allclose`.

- [ ] **Step 2: Run the new tests to verify they fail before additions**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/test_voxelnet.py tests/test_model_state.py`

Expected: FAIL because the state-prefix/checkpoint behavior is not yet asserted or because the required helper is absent.

- [ ] **Step 3: Add only the test helpers and documentation required for green**

Add `assert_task_maps_equal(actual, expected)` to `tests/test_voxelnet.py`; it must compare task count, ordered branch keys, shape, dtype, and values. Update docs to say “boundary-complete VoxelNet assembly implemented with injected sparse backend”; retain `SpMiddleResNetFHD`, CUDA parity, and official intermediate parity as pending.

- [ ] **Step 4: Run focused composition verification**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/test_voxelnet.py tests/test_model_state.py`

Expected: PASS.

- [ ] **Step 5: Commit composition evidence**

```bash
git add tests/test_voxelnet.py tests/test_model_state.py docs/UPSTREAM_MAP.md docs/NEXT_STEPS.md
git commit -m "test: lock VoxelNet composition contract"
```

### Task 4: Create The Executable Architecture Walkthrough

**Files:**
- Create: `notebooks/centerpoint_walkthrough.ipynb`
- Create: `tests/test_notebook_walkthrough.py`
- Modify: `notebooks/README.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: public `centerpoint.data`, `centerpoint.models`, `centerpoint.config`, and the test-only assembly pattern from Task 1.
- Produces: a self-contained, compact-output notebook that can be executed from a clean kernel after installing project test dependencies.

- [ ] **Step 1: Write failing notebook structure tests**

```python
def test_walkthrough_contains_all_architecture_sections():
    notebook = load_notebook("notebooks/centerpoint_walkthrough.ipynb")
    text = "\n".join(cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "markdown")
    for heading in (
        "Full Architecture", "Coordinates And Tasks", "Hard Voxels And Mean VFE",
        "Sparse Backbone Boundary", "RPN Neck", "Six-Task CenterHead",
        "Losses", "Dense Decode And NMS", "End-To-End Tensor Ledger",
    ):
        assert heading in text

def test_walkthrough_imports_package_and_has_no_large_outputs():
    notebook = load_notebook("notebooks/centerpoint_walkthrough.ipynb")
    code = "\n".join(cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert "from centerpoint" in code
    assert all(len(str(cell.get("outputs", []))) < 20_000 for cell in notebook["cells"])
```

Use standard-library `json` to load notebook JSON so tests do not add an `nbformat` dependency. Require markdown cells to contain ASCII arrows (`->`) and explicit `Input`, `Output`, and `Test` labels for all module-card headings.

- [ ] **Step 2: Run notebook tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/test_notebook_walkthrough.py`

Expected: FAIL because the notebook file does not exist.

- [ ] **Step 3: Create the notebook using valid notebook v4 JSON**

Create markdown/code cell pairs in this order:

```text
1. Full Architecture: ASCII graph and canonical tensor ledger.
2. Coordinates And Tasks: LiDAR axes, box fields, six task groups, target code order.
3. Data Preparation: sweep record, class balancing, augmentation, point/box contracts.
4. Hard Voxels And Mean VFE: voxel encounter order, zyx coordinates, VFE trace.
5. Sparse Backbone Boundary: SparseBackboneInput and deferred spconv contract.
6. RPN Neck: two blocks/deblocks and [B,256,H,W] -> [B,512,H,W].
7. Six-Task CenterHead: shared feature and six branch dictionaries.
8. Losses: focal and per-code gathered regression composition.
9. Dense Decode And NMS: all-cell decode, task NMS, offsets, merge.
10. End-To-End Tensor Ledger: inject a local notebook-only tiny SparseBackbone,
    construct VoxelNet through package APIs, run forward/loss/backward/predict,
    and print compact stage shapes and detection-field shapes.
```

Every markdown module card must include `Input:`, `Output:`, `I/O evolution:`, and `Test:` lines. Code uses only small spatial shape `(4,32,32)`, fixed seeds, and package imports. Keep notebook `outputs` empty in Git; the final cell is executable but not pre-executed.

- [ ] **Step 4: Add an optional clean-kernel execution command to the notebook test**

The test must always validate JSON/structure. If `jupyter` is available on `PATH`, execute:

```python
subprocess.run(
    ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
     "--ExecutePreprocessor.timeout=120", "notebooks/centerpoint_walkthrough.ipynb"],
    check=True,
)
```

After execution, reload the notebook and assert output text contains `Tensor ledger` and `gradient`. Clear outputs using notebook JSON before committing so Git stores no execution artifacts. If `jupyter` is absent, skip only execution with `pytest.skip`, while retaining structural validation.

- [ ] **Step 5: Run notebook-specific tests to verify green**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/test_notebook_walkthrough.py`

Expected: PASS, or one explicit skip only when `jupyter` is unavailable.

- [ ] **Step 6: Update notebook usage documentation**

Add this verified command to `notebooks/README.md` and the root README learning section:

```bash
jupyter notebook notebooks/centerpoint_walkthrough.ipynb
```

State that it is a boundary-complete VoxelNet walkthrough and that `SpMiddleResNetFHD` remains an injected, unimplemented production backend.

- [ ] **Step 7: Commit the notebook**

```bash
git add notebooks/centerpoint_walkthrough.ipynb notebooks/README.md README.md tests/test_notebook_walkthrough.py
git commit -m "docs: add CenterPoint architecture walkthrough"
```

### Task 5: Final Verification And Push

**Files:**
- Verify: all modified files from Tasks 1-4

**Interfaces:**
- Consumes: all implementation, tests, docs, and notebook from prior tasks.
- Produces: a clean, pushed atomic commit series with no unsupported reproduction claim.

- [ ] **Step 1: Run the full CPU suite**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q`

Expected: PASS with at most the existing optional backend/notebook execution skips.

- [ ] **Step 2: Inspect the complete commit series and worktree**

Run:

```bash
git status --short
git diff --check origin/main...HEAD
git log --oneline origin/main..HEAD
git diff --stat origin/main...HEAD
```

Expected: only the detector, tests, notebook, and scoped documentation are changed; the worktree is clean.

- [ ] **Step 3: Push the reviewed atomic commits**

```bash
git push origin main
```

- [ ] **Step 4: Verify remote synchronization**

Run: `git rev-list --left-right --count origin/main...HEAD`

Expected: `0 0`.

## Self-Review

Spec coverage: Task 2 implements the injected detector boundary and frozen state names; Task 3 locks state and behavior evidence; Task 4 implements the single, module-by-module architecture notebook and its execution contract; Task 5 verifies and pushes atomic commits. CUDA/spconv, full sparse implementation, export, and training remain explicitly out of scope.

Placeholder scan: no deferred work is assigned to a task; all deferred items are deliberate non-goals. Every code/testing step names a file, API, expected result, and command.

Type consistency: `VoxelNet`, `forward_features`, `loss`, and `predict` are defined once in Task 1 and used unchanged in Tasks 2-4. The test backbone implements existing `SparseBackbone.forward_sparse`; inference returns existing `Detections`.
