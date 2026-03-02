# Project setup

Use a project virtual environment so the tracker and Quarto run with consistent dependencies (and to avoid system Python/Homebrew conflicts).

## One-time setup

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`.venv` is already in `.gitignore`; it will not be committed.

## Running the comparables pipeline (monthly)

With the venv activated:

```bash
source .venv/bin/activate
python run_comparables.py --industry all
```

Or without activating: use `.venv/bin/python run_comparables.py --industry all`. Archived industry trackers (SEC/news/weekly): see `_archive/industry_trackers/README.md`. Or without activating, use the venv’s Python directly:

```bash
.venv/bin/python run_comparables.py --industry all
```

## Local site preview

From the repo root (with venv activated so Python chunks use the same env):

```bash
source .venv/bin/activate
quarto preview
```

Opens the site in your browser; live-reloads on file changes. The **Market Comps** page shows comparables tables from `outputs/comparables/` (run `python run_comparables.py --industry all` first to generate data).

## Quarto

Quarto will use whatever `python` is on your PATH. If you run `quarto preview` or `quarto render` with the venv activated, it will use the venv’s Python and the same packages. Everything else (rendering, site build) is unchanged.

## No impact on the rest of the project

- The site, Quarto, and existing notebooks work as before.
- The venv only affects how you run the tracker and which Python/packages are used when you run it or when Quarto runs Python chunks. You can still use your system or Anaconda Python for other work; use the venv only when working in this repo.
