"""Page assets: each one executes a jupytext-paired notebook (in `pages/`)
against energy-data-hub's current data and writes the executed `.ipynb`
into `book/notebooks/` for MyST to render.

Real, executable Dagster assets (not the non-executing `AssetSpec` marker
pattern `pecd-power-validity-DE/dagster_book_asset.py` used as a pilot --
see energy-data-hub/ARCHITECTURE.md's "stepping stone, not the final shape"
note). `deps=[...]` names hub assets by bare AssetKey string; resolving
across code locations works because `workspace.yaml` (in the hub repo)
loads both this repo and the hub together into one Asset Graph.

Materializing a page alone re-runs it against whatever's currently in the
hub's `data/` -- no upstream refresh. Materializing `+page_name` (Dagster's
upstream-selection syntax) first refreshes every upstream hub asset it
depends on, then the page.

**Materializing != publishing.** This only produces the executed `.ipynb`;
building the MyST site and deploying it to GitHub Pages is a separate,
manual step (`cd book && uv run myst build --html`, or push to `main` for
the GH Actions deploy) -- see book/markdown/index.md.
"""

import subprocess
from pathlib import Path

from dagster import AssetExecutionContext, MetadataValue, asset

REPO_ROOT = Path(__file__).resolve().parent.parent
PAGES_DIR = REPO_ROOT / "pages"
NOTEBOOKS_DIR = REPO_ROOT / "book" / "notebooks"

_PAGE_TAGS = {"load_pattern": "full_refresh"}


def _run_page(source_name: str, notebook_name: str) -> Path:
    """`jupytext --to notebook --execute` the page, then strip its jupytext
    metadata cell (MyST doesn't understand it) -- same two-step command
    already proven in mastr-power-capacities-germany's Makefile."""
    output_file = NOTEBOOKS_DIR / notebook_name
    subprocess.run(
        [
            "uv", "run", "jupytext", "--to", "notebook", "--execute",
            "--set-kernel", "python3", "--output", str(output_file),
            str(PAGES_DIR / source_name),
        ],
        check=True, cwd=REPO_ROOT,
    )
    subprocess.run(
        ["uv", "run", "python", str(PAGES_DIR / "_strip_jupytext_metadata.py"), str(output_file)],
        check=True, cwd=REPO_ROOT,
    )
    return output_file


@asset(
    name="page_mastr_capacity_de",
    deps=["mastr_capacity_events", "mastr_capacity_by_region_year", "nuts_regions", "country_borders", "offshore_regions"],
    group_name="pages",
    kinds={"notebook"},
    tags=_PAGE_TAGS,
    description=(
        "Installed wind + solar capacity: overview stats, growth over time, capacity maps by NUTS3 region, "
        "offshore footprint, and an animated year-by-year map. Stripped down from "
        "mastr-power-capacities-germany's 04_eda notebook -- no storage/PV-category sections yet, see "
        "book/markdown/index.md's roadmap."
    ),
)
def page_mastr_capacity_de(context: AssetExecutionContext) -> None:
    output_file = _run_page("01_mastr_capacity_de.py", "01_mastr_capacity_de.ipynb")
    context.add_output_metadata({"path": MetadataValue.path(str(output_file))})


@asset(
    name="page_mastr_vs_smard_capacity",
    deps=["mastr_capacity_by_region_year", "smard_capacity_solar", "smard_capacity_wind_onshore", "smard_capacity_wind_offshore"],
    group_name="pages",
    kinds={"notebook"},
    tags=_PAGE_TAGS,
    description=(
        "MaStR-derived vs. SMARD's own official installed-capacity figures, year by year, for solar and "
        "onshore/offshore wind -- a live sanity check on the MaStR-derived numbers used throughout this book, "
        "replacing a one-off hand-copied comparison against a single press release."
    ),
)
def page_mastr_vs_smard_capacity(context: AssetExecutionContext) -> None:
    output_file = _run_page("02_mastr_vs_smard_capacity.py", "02_mastr_vs_smard_capacity.ipynb")
    context.add_output_metadata({"path": MetadataValue.path(str(output_file))})


page_assets = [page_mastr_capacity_de, page_mastr_vs_smard_capacity]
