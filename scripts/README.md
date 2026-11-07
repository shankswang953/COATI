# Website maintenance

The public documentation is hosted at <https://shankswang953.github.io/COATI/>.
The source lives in `docs/` in the COATI repository; tutorial notebooks live in
`notebooks/`. Hosting does not require a separate repository or an OpenAI account.

## Build and preview locally

From the repository root, with the COATI environment activated:

```bash
python -m pip install -e '.[docs]'
python scripts/build_website.py --serve
```

Open <http://127.0.0.1:8000/>. The static website is written to
`docs/_build/html/`. Omit `--serve` to build without starting a preview server.

The build renders the saved notebook outputs. It does not run training, download
biological data or execute notebook cells. Run changed notebooks yourself before
publishing their new results.

## Publish

Push documentation or notebook changes to `main`. The **Publish documentation**
GitHub Actions workflow builds the website and deploys it to GitHub Pages.
It can also be started manually from the repository's Actions tab.

Repository setting: **Settings → Pages → Build and deployment → Source → GitHub Actions**.
The output includes `.nojekyll` so Sphinx assets such as `_static/` are served.
The GitHub Pages build uses the same `scripts/build_website.py` command as the
local preview. Generated HTML is ignored by Git and is uploaded as a deployment
artifact, rather than being mixed into the source tree.
