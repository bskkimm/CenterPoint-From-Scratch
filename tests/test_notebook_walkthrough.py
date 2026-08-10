import json
import shutil
import subprocess
from pathlib import Path

import pytest


NOTEBOOK_PATH = "notebooks/centerpoint_walkthrough.ipynb"
MODULE_CARD_HEADINGS = (
    "Full Architecture",
    "Coordinates And Tasks",
    "Hard Voxels And Mean VFE",
    "Sparse Backbone Boundary",
    "RPN Neck",
    "Six-Task CenterHead",
    "Losses",
    "Dense Decode And NMS",
    "End-To-End Tensor Ledger",
)


def load_notebook(path=NOTEBOOK_PATH):
    with open(path, encoding="utf-8") as notebook_file:
        return json.load(notebook_file)


def test_walkthrough_contains_all_architecture_sections():
    notebook = load_notebook()
    markdown_cells = [
        cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    ]
    text = "\n".join(markdown_cells)

    for heading in MODULE_CARD_HEADINGS:
        card = next(cell for cell in markdown_cells if heading in cell)
        assert "->" in card
        for label in ("Input:", "Output:", "I/O evolution:", "Test:"):
            assert label in card


def test_walkthrough_imports_package_and_has_no_large_outputs():
    notebook = load_notebook()
    code = "\n".join(
        cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "code"
    )

    assert "from centerpoint" in code
    assert all(cell.get("outputs", []) == [] for cell in notebook["cells"])


def test_walkthrough_documents_deferred_cuda_and_spconv_integration():
    notebook = load_notebook()
    notebook_text = "\n".join(
        cell["source"] for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    )

    for text in (
        Path("README.md").read_text(encoding="utf-8"),
        Path("notebooks/README.md").read_text(encoding="utf-8"),
        notebook_text,
    ):
        assert "boundary-complete" in text
        assert "SpMiddleResNetFHD" in text
        assert "CUDA" in text
        assert "spconv" in text
        assert "deferred" in text
        assert "unimplemented" in text


def test_walkthrough_executes_from_a_clean_kernel_when_jupyter_is_available(tmp_path):
    if shutil.which("jupyter") is None:
        pytest.skip("jupyter is not available on PATH")

    tracked_notebook = Path(NOTEBOOK_PATH)
    original_contents = tracked_notebook.read_bytes()
    execution_notebook = tmp_path / tracked_notebook.name
    execution_notebook.write_bytes(original_contents)

    try:
        subprocess.run(
            [
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                f"--output={execution_notebook.name}",
                "--ExecutePreprocessor.timeout=120",
                str(execution_notebook),
            ],
            check=True,
        )

        notebook = load_notebook(execution_notebook)
        output_text = "\n".join(
            output.get("text", "")
            for cell in notebook["cells"]
            for output in cell.get("outputs", [])
            if output.get("output_type") == "stream"
        )
        assert "Tensor ledger" in output_text
        assert "gradient" in output_text
    finally:
        assert tracked_notebook.read_bytes() == original_contents
