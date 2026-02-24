# The AI Strategist

This repository contains work for **The AI Strategist**, which applies artificial intelligence and data science to corporate strategy, private equity value creation, and capital allocation. We use machine learning and quantitative methods to surface strategic insights on revenue growth, operating efficiency, due diligence, forecasting, and competitive positioning.

Designed for investors, operators, and founders, this platform uses AI as the primary toolkit for strategy work, turning data into durable enterprise value across private markets and the broader economy.

Content is developed in Jupyter notebooks and published via [Quarto](https://quarto.org/).

---

## Setup

### Quarto CLI

Publishing uses the Quarto CLI. Install it first:

- **macOS (Homebrew):**
  ```bash
  brew install --cask quarto
  ```
- **Other systems:** See [Quarto -- Get Started](https://quarto.org/docs/get-started/) and choose your OS for installers and instructions.

Verify the installation:

```bash
quarto check
```

### Python / Jupyter

Use your preferred environment (venv, conda, etc.) and install Jupyter for authoring notebooks. The Quarto CLI will use your environment when rendering notebooks.

---

## Project layout

- **index.qmd** -- Homepage
- **blog.qmd** -- Essays listing (all posts, newest first)
- **about.qmd** -- About the platform
- **posts/** -- Blog posts (each post is a subfolder with `index.qmd`)
- **notebooks/** -- Jupyter notebooks (e.g. `notebooks/sample.ipynb`)
- **_quarto.yml** -- Quarto website config (navbar, theme, execute options)

---

## Publishing with Quarto

From the project root:

- Render the full site: `quarto render`
- Preview (watch and reload): `quarto preview`
- Render a single post: `quarto render posts/foundational-thesis/index.qmd`

Rendered output goes to `docs/`.

---

## Deploying to GitHub Pages

This site is configured for GitHub Pages using the `docs/` output directory.

1. Render the site locally:
   ```bash
   quarto render
   ```
2. Commit and push `docs/` along with your source changes:
   ```bash
   git add docs
   git commit -m "Render site"
   git push
   ```
3. In GitHub, go to **Settings > Pages** and set the source to **Deploy from a branch**, branch `main`, folder `/docs`.

The site will be available at `https://<username>.github.io/ai-strategist/`.

A `.nojekyll` file is included at the repo root so GitHub Pages does not apply Jekyll processing.
