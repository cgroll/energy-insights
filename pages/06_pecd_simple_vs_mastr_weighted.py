# ---
# jupytext:
#   text_representation:
#     format_name: percent
# kernelspec:
#   display_name: Python 3
#   language: python
#   name: python3
# ---

# %% [markdown]
# # DE capacity factors: three ways to weight PECD, compared
#
# `de_capacity_factor_current_fleet` (this hub's primary DE capacity-factor
# product) weights PECD's zone/technology-level capacity factors by MaStR's
# real, unit-level installed-capacity data -- accurate, but it only works
# where a MaStR-equivalent registry exists, i.e. Germany. This page asks: how
# far do we get with much simpler approximations that need no per-unit fleet
# data, or only a nationwide (not per-unit) slice of it?
#
# **Three variants compared, 2026-10-09:**
# 1. **`simple`** -- no MaStR at all (solar: fixed, hand-sourced external
#    market weights; wind: pure PECD zone-area geometry).
# 2. **`mastr_weighted` (new)** -- same wind geometry as `simple`, but
#    solar's technology mix now comes from real MaStR unit data (today's
#    actual installed-capacity split across the 4 technologies), not an
#    externally-sourced number. Built 2026-10-09 as
#    `de_capacity_factors_fleet_weighted` (`edh/pecd.py`), itself built on
#    top of the now-separate `de_technology_capacity_factors` (no MaStR) +
#    `de_fleet_weights_snapshot` (MaStR, timestamped) assets.
# 3. **`complex`** (`de_capacity_factor_current_fleet`) -- full per-unit MaStR
#    weighting at NUTS2 (solar) / zone-fraction (wind) resolution, the real
#    product.
#
# **This page's own prototype became a real hub asset.** The fixed-weight
# approximation below is no longer computed here -- it's read straight from
# `pecd_country_capacity_factors_simple` (`edh/pecd.py` /
# `edh_dagster/assets/pecd.py`), which generalized exactly this page's
# methodology to every PECD country, not just Germany. This page now only
# picks out the `DE` column and compares it against the real, MaStR-weighted
# `de_capacity_factor_current_fleet` -- the one country where that comparison
# is possible at all.
#
# ## The weights used, explicitly
#
# **Solar** -- PECD's `nuts_0` (true country-level) product already gives one
# series per technology (`60` industrial rooftop, `61` residential rooftop,
# `62` utility fixed-tilt, `63` utility tracking) with no region aggregation
# needed. The open question is only how to blend these four into one number.
# Fixed weights below, held constant across the whole history (not MaStR, but
# not equal-weighted either -- equal weights would hand utility-tracking a
# quarter of the mix, when it's close to absent in Germany):
#
# | technology | weight | basis |
# |---|---|---|
# | `62` utility fixed-tilt | 31% | ~40 of ~120 GW total DE PV capacity was ground-mounted at end of 2025 (pv magazine, citing BSW-Solar/BNetzA), minus tracking's sliver |
# | `63` utility tracking | 2% | only 23 of Germany's 300 largest solar parks use tracking (rare, uneconomical at today's module prices) |
# | `61` residential rooftop | 39% | 2025 additions: 5.2 GW private roof vs. 3.7 GW large commercial roof (BSW-Solar) -> ~58/42 split of the ~67% rooftop share |
# | `60` industrial/commercial rooftop | 28% | the remaining ~42% of the rooftop share |
#
# **⚠️ These are DE-market weights, and `pecd_country_capacity_factors_simple`
# currently applies them to every PECD country, not just Germany** -- no
# equivalent per-country technology-mix data has been sourced yet (checked
# 2026-09-24: SolarPower Europe's country segment tables are member-only, and
# nothing at all splits utility fixed-tilt vs. tracking by country). That's a
# real, potentially large distortion for any *other* country's solar column in
# that hub asset -- see its `data_quality_warning` metadata and
# `energy-data-hub/README.md`'s "Known data-quality caveats". Not an issue for
# this page, which only ever looks at `DE`.
#
# **Wind onshore/offshore** -- PECD has no `nuts_0` product for wind at all
# (confirmed against the live CDS API, see
# `energy-data-hub/docs/pecd_data_availability.md`), only zone-level
# (`PEON`/`PEOF`). The simple approximation here: weight each zone by its
# physical area (from PECD's own rasterized zone mask), not by installed
# capacity -- i.e. assume turbines are spread uniformly per km² rather than
# concentrated by real siting. Zones PECD never modeled (100% NaN capacity
# factor) are dropped and the remaining zones' weights renormalized to sum to
# 1 -- this generalizes cleanly per country since it's real geometry, not a
# borrowed default (unlike solar above).

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from insights.paths import hub_file

COMPLEX_COLOR = "#2a78d6"  # de_capacity_factor_current_fleet -- full MaStR weighting, this hub's real product
SIMPLE_COLOR = "#eb6834"  # pecd_country_capacity_factors_simple -- no MaStR involved
MASTR_WEIGHTED_COLOR = "#2a9d6b"  # de_capacity_factors_fleet_weighted -- nationwide MaStR solar mix, no NUTS2/zone detail

# %% [markdown]
# ## The real MaStR solar technology mix, today
#
# What `mastr_weighted` actually uses instead of the hand-sourced external
# weights above -- read straight from `de_fleet_weights_snapshot`, not
# recomputed here.

# %%
fleet_weights = pd.read_parquet(hub_file("pecd", "de_fleet_weights_snapshot.parquet")).iloc[0]
weight_comparison = pd.DataFrame({
    "external (simple)": {"60": 0.28, "61": 0.39, "62": 0.31, "63": 0.02},
    "MaStR, today (mastr_weighted)": {
        "60": fleet_weights["solar_weight_60"], "61": fleet_weights["solar_weight_61"],
        "62": fleet_weights["solar_weight_62"], "63": fleet_weights["solar_weight_63"],
    },
})
print(f"MaStR fleet snapshot as of {fleet_weights['as_of'].date()}:")
print(weight_comparison.round(3))

# %% [markdown]
# ## DE columns from `pecd_country_capacity_factors_simple`

# %%
country_simple = pd.read_parquet(hub_file("pecd", "pecd_country_capacity_factors_simple.parquet"))
de_simple = country_simple.xs("DE", axis=1, level="country")

solar_simple = de_simple["solar"].rename("simple")
onshore_simple = de_simple["wind_onshore"].rename("simple")
offshore_simple = de_simple["wind_offshore"].rename("simple")

# %% [markdown]
# ## Comparison against `de_capacity_factor_current_fleet`
#
# Headline numbers below are computed **at native hourly resolution** --
# every one of the ~96,400 hours 2015-2025, not the monthly/annual
# aggregates the charts further down use for readability. Aggregating to
# monthly (or coarser) smooths out a lot of hour-to-hour noise, so a
# correlation/MAE computed on monthly means alone would overstate how well
# the simpler approximations track the real thing *within* a month -- shown
# explicitly in the resolution-comparison table right after.

# %%
complex_cf = pd.read_parquet(hub_file("pecd", "de_capacity_factor_current_fleet.parquet"))
mastr_weighted_cf = pd.read_parquet(hub_file("pecd", "de_capacity_factors_fleet_weighted.parquet"))

SERIES = {
    "solar": {
        "simple": solar_simple,
        "mastr_weighted": mastr_weighted_cf["capacity_factor_solar"],
        "complex": complex_cf["capacity_factor_solar"],
    },
    "wind_onshore": {
        "simple": onshore_simple,
        "mastr_weighted": mastr_weighted_cf["capacity_factor_wind_onshore"],
        "complex": complex_cf["capacity_factor_wind_onshore"],
    },
    "wind_offshore": {
        "simple": offshore_simple,
        "mastr_weighted": mastr_weighted_cf["capacity_factor_wind_offshore"],
        "complex": complex_cf["capacity_factor_wind_offshore"],
    },
}
VARIANTS = ("simple", "mastr_weighted")  # each compared against "complex"


def error_stats(variant: pd.Series, complex_: pd.Series, resample: str | None = None) -> dict:
    """Deviation metrics between one approximation and `complex` (the real,
    full MaStR-weighted product). `resample=None` keeps the native hourly
    resolution; e.g. `resample="D"` or `"MS"` averages both series to that
    frequency first -- so the same function makes the resolution-sensitivity
    comparison below an apples-to-apples one-liner."""
    df = pd.concat([variant.rename("variant"), complex_.rename("complex")], axis=1, sort=False).dropna()
    if resample is not None:
        df = df.resample(resample).mean()
    err = df["variant"] - df["complex"]
    return {
        "n_obs": len(df),
        "mean_variant": df["variant"].mean(),
        "mean_complex": df["complex"].mean(),
        "corr": df["variant"].corr(df["complex"]),
        "bias_pp": err.mean() * 100,
        "mae_pp": err.abs().mean() * 100,
        "rmse_pp": (err**2).mean() ** 0.5 * 100,
    }


stats = pd.concat(
    {
        variant: pd.DataFrame({tech: error_stats(series[variant], series["complex"]) for tech, series in SERIES.items()}).T
        for variant in VARIANTS
    },
    axis=0,
)
print("Hourly deviation metrics (native resolution), each variant vs. complex:")
print(stats.round(4))

# %% [markdown]
# ## Does resolution change the picture? Hourly vs. daily vs. monthly

# %%
resolutions = {"hourly": None, "daily": "D", "monthly": "MS"}
by_resolution = pd.concat(
    {
        variant: pd.concat(
            {
                tech: pd.DataFrame(
                    {label: error_stats(series[variant], series["complex"], resample=freq) for label, freq in resolutions.items()}
                ).T
                for tech, series in SERIES.items()
            },
            axis=0,
        )
        for variant in VARIANTS
    },
    axis=0,
)
print(by_resolution[["n_obs", "corr", "mae_pp", "rmse_pp"]].round(4))

# %% [markdown]
# ## Monthly capacity factor, full history, per technology

# %%
fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
for ax, (tech, series) in zip(axes, SERIES.items()):
    df = pd.concat(
        {variant: series[variant] for variant in VARIANTS} | {"complex": series["complex"]}, axis=1
    ).dropna()
    monthly = df.resample("MS").mean()
    ax.plot(monthly.index, monthly["complex"], label="Complex (de_capacity_factor_current_fleet)", color=COMPLEX_COLOR, linewidth=1.3)
    ax.plot(monthly.index, monthly["simple"], label="Simple (external weights, no MaStR)", color=SIMPLE_COLOR, linewidth=1.3, linestyle="--")
    ax.plot(monthly.index, monthly["mastr_weighted"], label="MaStR-weighted (nationwide mix, no NUTS2/zone detail)", color=MASTR_WEIGHTED_COLOR, linewidth=1.3, linestyle=":")
    ax.set_ylabel("Capacity factor")
    ax.set_title(tech)
    ax.legend(loc="upper left", fontsize=8)
fig.suptitle("DE capacity factor, monthly mean: three variants")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Annual mean, per technology

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (tech, series) in zip(axes, SERIES.items()):
    df = pd.concat(
        {variant: series[variant] for variant in VARIANTS} | {"complex": series["complex"]}, axis=1
    ).dropna()
    annual = df.groupby(df.index.year).mean()
    x = np.arange(len(annual))
    width = 0.26
    ax.bar(x - width, annual["complex"], width, label="Complex", color=COMPLEX_COLOR)
    ax.bar(x, annual["simple"], width, label="Simple", color=SIMPLE_COLOR)
    ax.bar(x + width, annual["mastr_weighted"], width, label="MaStR-weighted", color=MASTR_WEIGHTED_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(annual.index, rotation=45)
    ax.set_ylabel("Capacity factor")
    ax.set_title(tech)
    ax.legend(fontsize=8)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Takeaways
#
# - **Solar: `mastr_weighted` does not beat `simple`, surprisingly.**
#   `simple`'s external, hand-sourced weights score corr 0.9993 / MAE 0.302pp
#   against `complex`; swapping in `mastr_weighted`'s real, today-dated MaStR
#   technology mix (60/61/62/63 = 33.0/32.6/34.0/0.4%, vs. the external
#   28/39/31/2%) actually scores very slightly *worse* (corr 0.9993 / MAE
#   0.309pp). Reason: both `simple` and `mastr_weighted` apply **one single
#   nationwide** technology-mix percentage, while `complex` varies the mix by
#   NUTS2 region -- real regional differences in rooftop-vs-utility share
#   aren't captured by a nationwide number either way, so getting that one
#   number right from real data (`mastr_weighted`) vs. a reasonable published
#   estimate (`simple`) barely matters next to the missing regional detail.
# - **Wind onshore: `mastr_weighted` is numerically identical to `simple`.**
#   Both use the exact same PECD PEON zone-area geometry -- MaStR plays no
#   role in either one's wind weighting, by design (see module docstring).
#   Corr 0.975, MAE 3.16pp either way, unchanged from before.
# - **Wind offshore: `mastr_weighted` is a genuine methodology upgrade that
#   nonetheless scores slightly worse.** `mastr_weighted` is the first variant
#   to actually *area-weight* the coarser `p2of` zones (a `p2of` mask was
#   downloaded 2026-10-08, specifically to enable this), where `simple` still
#   falls back to an unweighted `p2of` mean for lack of that mask. Despite
#   being the more principled calculation, it scores corr 0.981 / MAE 4.33pp
#   vs. `simple`'s corr 0.986 / MAE 3.78pp -- a real, if modest, example of
#   area-weighting *not* helping on this particular zone scheme, consistent
#   with onshore's own "7 zones don't differ enough in area to matter" finding
#   below.
# - **Net: all three variants sit within ~1 correlation point of `complex`
#   across the board** -- solar 0.999 regardless of weighting choice, wind
#   0.975-0.986 regardless of whether the nationwide weight is external,
#   MaStR-derived, or area-weighted vs. not. For everything built on top of
#   this hub (the long DE climatology, the WeatherNext grid reports, etc.),
#   `mastr_weighted` is a reasonable one to standardize on precisely because
#   it's grounded in real MaStR data without needing NUTS2/zone-fraction
#   detail -- not because it scores measurably better than the alternatives,
#   it mostly doesn't.
# - **Resolution matters a lot for how good any of this looks**: the same
#   comparison at monthly-mean resolution flatters both wind variants --
#   correlation climbs from ~0.975-0.986 (hourly) to ~0.994-0.996 (monthly),
#   and MAE drops roughly 3-4x. Averaging a whole month together cancels out
#   a lot of hour-to-hour disagreement that's real at the resolution most
#   uses (e.g. Dunkelflaute analysis) actually care about -- the hourly
#   numbers in the headline table above, not the monthly chart further up,
#   are the honest measure of how well any approximation tracks reality.
# - **Caveat that doesn't show up in these numbers:** wind's zone geometry
#   generalizes cleanly to every PECD country (real area, not a borrowed
#   default); solar's technology-mix weights -- external *and*
#   MaStR-derived -- are Germany-specific and only validated here, for
#   Germany. `pecd_country_capacity_factors_simple` still reuses Germany's
#   external weights for every other country; nothing about this page's
#   `mastr_weighted` finding extends that validation elsewhere.
