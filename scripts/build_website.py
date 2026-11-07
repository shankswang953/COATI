"""Build the documentation from saved notebook outputs and optionally preview it."""

import argparse
import functools
import http.server
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", help="Preview the built website locally")
    parser.add_argument("--port", type=int, default=8000, help="Local preview port")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / "docs" / "_build" / "html"
    env = {**os.environ, "LC_ALL": "C", "LANG": "C"}
    subprocess.run([sys.executable, "scripts/build_notebook_pages.py"], cwd=root,
                   env=env, check=True)
    subprocess.run([sys.executable, "-m", "sphinx", "-W", "--keep-going", "-b", "html",
                    "docs", str(output)], cwd=root, env=env, check=True)
    (output / ".nojekyll").touch()
    print(f"Website built: {output}", flush=True)
    if args.serve:
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(output))
        with http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
            print(f"Local preview: http://127.0.0.1:{args.port}/ (Ctrl+C to stop)", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
