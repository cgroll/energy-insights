# Energy Insights

Explanatory pages on the German energy system — installed capacity, how it's
grown, where it sits, how well official registries agree, and (eventually)
what today's fleet might mean for past Dunkelflauten. Published as a
[MyST](https://mystmd.org/) book.

Owns no data of its own — every page reads directly from
[`energy-data-hub`](https://github.com/cgroll/energy-data-hub), the shared
Dagster-native ingestion layer for MaStR/SMARD/PECD data. See that repo's
`ARCHITECTURE.md` for the overall design (this is "the combined book").

## Layout

```
insights/              # plain Python package: shared helpers
  paths.py             # resolves paths into ../energy-data-hub/data/
insights_dagster/       # Dagster layer: one real, executable asset per page
  assets.py             # page_mastr_capacity_de, ...
  definitions.py         # Definitions() entry point
pages/                  # jupytext-paired (percent format) notebook sources
  01_mastr_capacity_de.py
book/
  myst.yml
  markdown/index.md
  notebooks/            # executed .ipynb output, committed
```

## Running it

Requires `energy-data-hub` cloned as a sibling repo with the relevant assets
already materialized (`mastr_capacity_by_region_year`, `nuts_regions`,
`country_borders`, `offshore_regions`, ...).

```bash
uv sync
export DAGSTER_HOME=~/research/energy-data-hub/.dagster_home  # shared with the hub
uv run dagster asset materialize --select page_mastr_capacity_de -f insights_dagster/definitions.py
```

Or, to see this repo's pages in the same Asset Graph as the hub's data (and
refresh a page's upstream hub dependencies first via `+page_name`):

```bash
cd ~/research/energy-data-hub
uv run dagster dev -w workspace.yaml
```

Materializing a page only produces the executed notebook. To actually
publish the site:

```bash
cd book && uv run myst build --html   # or: uv run myst start, for live preview
```

Pushing to `main` also builds + deploys to GitHub Pages via
`.github/workflows/deploy.yml` — but only from whatever `.ipynb` files are
already committed in `book/notebooks/`, so publishing is always a deliberate
step, never an implicit side effect of a hub data refresh.
