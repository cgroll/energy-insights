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
# # MaStR vs. SMARD: installed capacity
#
# The Bundesnetzagentur publishes its own official year-end installed-
# capacity figures via SMARD, independently of the unit-level MaStR
# register. Comparing the two is a useful sanity check on the MaStR-derived
# numbers used throughout the rest of this book -- unlike a one-off,
# hand-copied comparison against a single year's press release, this reads
# both sides live from `energy-data-hub` so it stays current automatically.

# %%
import matplotlib.pyplot as plt
import pandas as pd

from insights.paths import hub_file

panel = pd.read_parquet(hub_file("capacity", "capacity_by_region_year.parquet"))

TECHNOLOGIES = ["solar", "wind_onshore", "wind_offshore"]
TECHNOLOGY_LABELS = {"solar": "Solar", "wind_onshore": "Wind (onshore)", "wind_offshore": "Wind (offshore)"}
TECHNOLOGY_COLORS = {"solar": "#E8A33D", "wind_onshore": "#1B7A9C", "wind_offshore": "#8B4A1E"}

# %% [markdown]
# ## MaStR side: national annual totals
#
# Summed across all regions per year; wind is split into onshore/offshore
# using the same `DEZZ` offshore pseudo-region convention used elsewhere in
# this book. Includes the current (still-incomplete) year -- MaStR's
# "year-end" snapshot for it is really just "as of today".

# %%
is_offshore = panel["region_code"].str.startswith("DEZZ")
mastr_national = pd.DataFrame({
    "solar": panel[panel["technology"] == "solar"].groupby("year")["capacity_mw"].sum(),
    "wind_onshore": panel[(panel["technology"] == "wind") & ~is_offshore].groupby("year")["capacity_mw"].sum(),
    "wind_offshore": panel[(panel["technology"] == "wind") & is_offshore].groupby("year")["capacity_mw"].sum(),
})
mastr_national.tail()

# %% [markdown]
# ## SMARD side: official year-end capacity
#
# SMARD's own capacity series are published hourly but only actually change
# value a handful of times a year -- effectively a year-end (Dec 31)
# snapshot repeated forward until the next update, the same cadence as
# MaStR's annual panel above.

# %%
smard_national = pd.DataFrame({
    tech: pd.read_parquet(hub_file("smard", f"capacity_{tech}.parquet")).resample("YE").last().iloc[:, 0]
    for tech in TECHNOLOGIES
})
smard_national.index = smard_national.index.year
smard_national.tail()

# %% [markdown]
# ## Current installed capacity
#
# Where things stand *right now*: MaStR's live snapshot next to SMARD's most
# recently published figure. These two are **not** a like-for-like
# comparison -- SMARD only updates a handful of times a year, so its number
# can lag today's real installed capacity by close to a year. This section
# answers "how much is installed today", not "do the two sources agree"
# (that's the next section, restricted to a year both sides have actually
# reported).

# %%
today = pd.Timestamp.today()
smard_last_change = {}
smard_last_value = {}
for tech in TECHNOLOGIES:
    raw = pd.read_parquet(hub_file("smard", f"capacity_{tech}.parquet")).iloc[:, 0]
    changes = raw[raw.diff().ne(0)]
    smard_last_change[tech] = changes.index[-1].date()
    smard_last_value[tech] = changes.iloc[-1]

current = pd.DataFrame({
    "mastr_mw_today": mastr_national.iloc[-1],
    "smard_mw_last_published": pd.Series(smard_last_value),
    "smard_last_updated": pd.Series(smard_last_change),
})
print(f"MaStR as of {today.date()} vs. SMARD's own last real update per technology:")
print(current)

# %% [markdown]
# ## Evolution over time

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
for ax, tech in zip(axes, TECHNOLOGIES):
    ax.plot(mastr_national.index, mastr_national[tech], marker="o", label="MaStR", color=TECHNOLOGY_COLORS[tech])
    ax.plot(smard_national.index, smard_national[tech], marker="s", linestyle="--", label="SMARD", color="#444444")
    ax.set_title(TECHNOLOGY_LABELS[tech])
    ax.set_xlabel("Year")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Installed capacity (MW)")
axes[0].legend(loc="upper left", frameon=False)
fig.suptitle("MaStR vs. SMARD installed capacity, full history")
fig.tight_layout()
plt.show()

# %% [markdown]
# MaStR's line runs one point further right than SMARD's -- that's the
# current, still-incomplete year discussed above, not a data problem.

# %% [markdown]
# ## Do the two sources agree? (latest fully completed year)
#
# The current year is excluded here: MaStR's snapshot for it means
# "capacity as of today", while SMARD's own figure for it is just last
# year's number carried forward until they publish a real update --
# including it would make MaStR look artificially ahead by however many
# months have elapsed this year, not because it actually is.

# %%
current_year = today.year
common_years = mastr_national.index.intersection(smard_national.index)
completed_years = common_years[common_years < current_year]

latest_completed_year = completed_years.max()
comparison = pd.DataFrame({
    "mastr_mw": mastr_national.loc[latest_completed_year],
    "smard_mw": smard_national.loc[latest_completed_year],
})
comparison["diff_mw"] = comparison["mastr_mw"] - comparison["smard_mw"]
comparison["diff_pct"] = 100 * comparison["diff_mw"] / comparison["smard_mw"]
print(f"Year: {latest_completed_year}")
print(comparison.round(1))

# %% [markdown]
# Onshore and offshore wind typically match SMARD closely (within a few
# percent). Solar's match varies more year to year -- a known
# characteristic of MaStR rather than a data-pipeline bug: it's a
# self-reported registry, and small rooftop PV systems in particular are
# often registered with a lag of months, so the most recent year is always
# somewhat under- or overcounted until later backfilling catches up (see
# the [open-mastr data quality paper](https://doi.org/10.1145/3717413.3717421)
# referenced in its docs).

# %% [markdown]
# ## Full history, side by side

# %%
full_comparison = pd.concat(
    {tech: pd.DataFrame({"mastr_mw": mastr_national.loc[completed_years, tech], "smard_mw": smard_national.loc[completed_years, tech]}) for tech in TECHNOLOGIES},
    axis=1,
)
full_comparison.round(0)
