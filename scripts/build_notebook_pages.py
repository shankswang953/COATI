"""Prepare native notebook tutorial pages for Sphinx; optionally execute them first.

python scripts/build_notebook_pages.py --execute --kernel python3
python -m sphinx -W --keep-going -b html docs docs/_build/html
"""
import argparse
from pathlib import Path
import nbformat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--kernel", default="python3")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / "docs/tutorials"
    out.mkdir(parents=True, exist_ok=True)
    for source in sorted((root / "notebooks").glob("*.ipynb")):
        notebook = nbformat.read(source, as_version=4)
        if args.execute:
            from nbclient import NotebookClient
            NotebookClient(notebook, timeout=600, kernel_name=args.kernel,
                           resources={"metadata": {"path": str(root)}}).execute()
            nbformat.write(notebook, source)
        # Keep web titles short and retain all code and scientific outputs.
        titles = {"01_gaussian": "Gaussian translation", "02_paired": "Paired modalities",
                  "03_split": "Splitting and growth", "04_inputs": "Input adapters",
                  "05_velocity": "Vector fields and UMAP velocity",
                  "06_heldout_biological": "Held-out biological reconstruction",
                  "07_dynamics": "Cell types, growth and UMAP flow"}
        if notebook.cells and notebook.cells[0].cell_type == "markdown":
            first = notebook.cells[0].source.splitlines()
            first[0] = "# " + titles.get(source.stem, first[0].lstrip("# "))
            notebook.cells[0].source = "\n".join(first)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                # Optional widget availability does not affect tutorial results.
                cell.outputs = [output for output in cell.get("outputs", [])
                                if not (output.output_type == "stream"
                                        and output.get("name") == "stderr"
                                        and "IProgress not found" in output.get("text", ""))]
        # MyST-NB renders each notebook within the documentation theme.
        nbformat.write(notebook, out / source.name)
        print(source.name)


if __name__ == "__main__":
    main()
