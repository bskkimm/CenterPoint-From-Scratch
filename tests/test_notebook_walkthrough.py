import json
import shutil
import subprocess

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


def test_walkthrough_executes_from_a_clean_kernel_when_jupyter_is_available():
    if shutil.which("jupyter") is None:
        pytest.skip("jupyter is not available on PATH")

    try:
        subprocess.run(
            [
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--inplace",
                "--ExecutePreprocessor.timeout=120",
                NOTEBOOK_PATH,
            ],
            check=True,
        )

        notebook = load_notebook()
        output_text = "\n".join(
            output.get("text", "")
            for cell in notebook["cells"]
            for output in cell.get("outputs", [])
            if output.get("output_type") == "stream"
        )
        assert "Tensor ledger" in output_text
        assert "gradient" in output_text
    finally:
        notebook = load_notebook()
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                cell["execution_count"] = None
                cell["outputs"] = []
        with open(NOTEBOOK_PATH, "w", encoding="utf-8") as notebook_file:
            json.dump(notebook, notebook_file, indent=1)
            notebook_file.write("\n")
