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


@asset(
    name="page_pv_categories",
    deps=["mastr_capacity_events", "mastr_capacity_by_region_year_pv_category", "nuts_regions"],
    group_name="pages",
    kinds={"notebook"},
    tags=_PAGE_TAGS,
    description=(
        "Behind-the-meter PV categories (full grid feed-in / self-consumption with or without storage): how the "
        "mix has grown over time, whether it's similar across German states, plant-size distribution per "
        "category, and a cross-check against usage sector / installation type."
    ),
)
def page_pv_categories(context: AssetExecutionContext) -> None:
    output_file = _run_page("03_pv_categories.py", "03_pv_categories.ipynb")
    context.add_output_metadata({"path": MetadataValue.path(str(output_file))})


@asset(
    name="page_pecd_potential_vs_smard_observed",
    deps=["de_potential_historic", "mastr_capacity_by_region_year", "smard_generation_solar", "smard_generation_wind_onshore", "smard_generation_wind_offshore"],
    group_name="pages",
    kinds={"notebook"},
    tags=_PAGE_TAGS,
    description=(
        "PECD-derived renewable potential (de_potential_historic) vs. SMARD's actually observed generation: "
        "headline error stats, monthly-mean overlay, an hourly zoom-in, and capacity-factor/GW scatter views. "
        "Mirrors pecd-power-validity-DE's potential-vs-observed analysis against the hub's own potential panel."
    ),
)
def page_pecd_potential_vs_smard_observed(context: AssetExecutionContext) -> None:
    output_file = _run_page("04_pecd_potential_vs_smard_observed.py", "04_pecd_potential_vs_smard_observed.ipynb")
    context.add_output_metadata({"path": MetadataValue.path(str(output_file))})


@asset(
    name="page_pecd_simple_vs_mastr_weighted",
    deps=["pecd_country_capacity_factors_simple", "de_capacity_factor_current_fleet"],
    group_name="pages",
    kinds={"notebook"},
    tags=_PAGE_TAGS,
    description=(
        "DE columns of pecd_country_capacity_factors_simple (MaStR-free approximation: solar via fixed "
        "publicly-sourced technology-mix weights, wind onshore/offshore via PECD zone-mask area weights) "
        "compared against de_capacity_factor_current_fleet's real MaStR-weighted series -- the one country "
        "where that comparison is possible. This page prototyped the hub asset's methodology; it now reads "
        "the asset's output rather than recomputing it. See energy-data-hub/docs/pecd_data_availability.md."
    ),
)
def page_pecd_simple_vs_mastr_weighted(context: AssetExecutionContext) -> None:
    output_file = _run_page("06_pecd_simple_vs_mastr_weighted.py", "06_pecd_simple_vs_mastr_weighted.ipynb")
    context.add_output_metadata({"path": MetadataValue.path(str(output_file))})


@asset(
    name="page_ttf_gas_vs_power_price",
    deps=["ttf_gas_price", "smard_price_de_lu"],
    group_name="pages",
    kinds={"notebook"},
    tags=_PAGE_TAGS,
    description=(
        "Monthly average TTF gas price vs. SMARD's DE-LU day-ahead power price -- both in EUR/MWh, plotted on "
        "one shared axis. Full-history overlay, a correlation scatter, and the power-minus-gas spread over "
        "time, as a first look at how closely gas tracks the merit-order price-setting story in Germany."
    ),
)
def page_ttf_gas_vs_power_price(context: AssetExecutionContext) -> None:
    output_file = _run_page("07_ttf_gas_vs_power_price.py", "07_ttf_gas_vs_power_price.ipynb")
    context.add_output_metadata({"path": MetadataValue.path(str(output_file))})


@asset(
    name="page_pecd_country_comparison",
    deps=["pecd_country_capacity_factors_simple", "pecd_wind_offshore_europe_capacity_factors", "peof_region_mask"],
    group_name="pages",
    kinds={"notebook"},
    tags=_PAGE_TAGS,
    description=(
        "Cross-country long-run mean capacity factors for solar PV, wind onshore, and wind offshore: horizontal "
        "bar charts, a solar-vs-onshore-wind complementarity scatter, and choropleth maps -- replicating "
        "world-of-energy's 37_analyse_pecd page against this hub's own pecd_country_capacity_factors_simple. "
        "Offshore is mapped per individual p2of zone (rasterized from the peof zone mask) rather than "
        "collapsed to one color per country, to show within-country variation (e.g. North Sea vs. Baltic)."
    ),
)
def page_pecd_country_comparison(context: AssetExecutionContext) -> None:
    output_file = _run_page("08_pecd_country_comparison.py", "08_pecd_country_comparison.ipynb")
    context.add_output_metadata({"path": MetadataValue.path(str(output_file))})


page_assets = [
    page_mastr_capacity_de,
    page_mastr_vs_smard_capacity,
    page_pv_categories,
    page_pecd_potential_vs_smard_observed,
    page_pecd_simple_vs_mastr_weighted,
    page_ttf_gas_vs_power_price,
    page_pecd_country_comparison,
]
