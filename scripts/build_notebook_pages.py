"""Export tutorial notebooks for the documentation; optionally execute them first.

python scripts/build_notebook_pages.py --execute --kernel python3
python -m sphinx -W --keep-going -b html docs docs/_build/html
"""
import argparse
from pathlib import Path
import shutil
import nbformat
from nbconvert import HTMLExporter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--kernel", default="python3")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / "docs/_static/notebooks"
    out.mkdir(parents=True, exist_ok=True)
    for source in sorted((root / "notebooks").glob("*.ipynb")):
        notebook = nbformat.read(source, as_version=4)
        if args.execute:
            from nbclient import NotebookClient
            NotebookClient(notebook, timeout=600, kernel_name=args.kernel,
                           resources={"metadata": {"path": str(root)}}).execute()
            nbformat.write(notebook, source)
        html, _ = HTMLExporter(template_name="lab").from_notebook_node(notebook)
        (out / (source.stem + ".html")).write_text(html)
        shutil.copyfile(source, out / source.name)
        print(source.name)


if __name__ == "__main__":
    main()
