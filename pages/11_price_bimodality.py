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
# # From one daily price peak to two?
#
# The day-ahead price used to have roughly one daily shape: low overnight,
# rising through the morning, staying elevated through the day, peaking in
# the evening. The claim worth checking is that this has changed -- as PV
# capacity grew (see `10_pv_capture_rate`), midday prices get pushed down
# while the plants still setting the price in the morning and evening ramp
# don't, splitting one broad daytime peak into two separate ones either side
# of a solar-driven midday trough.
#
# That shape matters directly for battery arbitrage: a battery wants a big
# spread between its cheapest and most expensive hours *and* a well-defined
# trough to charge into, ideally twice a day rather than once. This page
# checks three things against SMARD's DE-LU day-ahead price: (1) has the
# daily min-max spread grown, (2) same question at monthly/annual
# resolution where a trend is actually readable, and (3) has the average
# daily *shape* itself become more two-humped -- checked separately for
# summer and winter, since PV-driven midday suppression is a summer story
# and mixing the two seasons would just wash it out.

# %%
import io

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from insights.paths import INSIGHTS_ROOT, hub_file

MIN_COLOR = "#2a78d6"     # dataviz skill categorical slot 1 (blue) -- cool = daily low
MAX_COLOR = "#eb6834"     # dataviz skill categorical slot 2 (orange) -- warm = daily high
SPREAD_COLOR = "#1baf7a"  # dataviz skill categorical slot 3 (aqua)
WINTER_CMAP = "Blues"     # cool season -> blue sequential ramp
SUMMER_CMAP = "Oranges"   # warm season -> orange sequential ramp, the skill's own
                          # rule for a second simultaneous sequential context

price = pd.read_parquet(hub_file("smard", "price_de_lu.parquet"))
price.tail()

# %% [markdown]
# ## Daily min vs. max: does a full-history plot even work?
#
# The most direct version of the question -- one point (or band) per day,
# ~2,900 of them. The current, still-incomplete day is dropped.

# %%
current_day = pd.Timestamp.today().normalize()

daily = pd.DataFrame({
    "min": price["price_de_lu"].resample("D").min(),
    "max": price["price_de_lu"].resample("D").max(),
})
daily = daily[daily.index < current_day]
daily["spread"] = daily["max"] - daily["min"]
daily.tail()

# %%
fig, ax = plt.subplots(figsize=(12, 5))
ax.fill_between(daily.index, daily["min"], daily["max"], color=SPREAD_COLOR, alpha=0.25, linewidth=0)
ax.plot(daily.index, daily["min"], color=MIN_COLOR, linewidth=0.6)
ax.plot(daily.index, daily["max"], color=MAX_COLOR, linewidth=0.6)
ax.set_ylabel("EUR/MWh")
ax.set_xlabel("Day")
ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Daily min/max day-ahead price, full history")
fig.tight_layout()
plt.show()

# %% [markdown]
# Readable as a "does the band widen" check -- and it clearly does, 2022's
# crisis aside -- but daily noise dominates any within-band detail, and a
# single wide day here and there tells you nothing about whether that's
# becoming typical. Monthly and annual averages of the same `spread` column
# are the better view for an actual trend.

# %% [markdown]
# ## Monthly and annual average spread
#
# `spread` averaged per calendar month, then per year -- same
# current-period exclusion as `07_ttf_gas_vs_power_price` and
# `10_pv_capture_rate`. 2018 is dropped from the annual view too: the price
# series only starts 2018-09-30, so a 2018 bucket would average three
# unrepresentative autumn months against full years everywhere else.

# %%
current_month = pd.Timestamp.today().to_period("M")
monthly_spread = daily["spread"].resample("MS").mean()
monthly_spread = monthly_spread[monthly_spread.index.to_period("M") < current_month]

fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(monthly_spread.index, monthly_spread, color=SPREAD_COLOR, linewidth=2)
ax.set_ylabel("EUR/MWh")
ax.set_xlabel("Month")
ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Average daily min-max spread, monthly")
fig.tight_layout()
plt.show()

# %%
current_year = pd.Timestamp.today().to_period("Y")
annual_spread = daily["spread"].resample("YS").mean()
annual_spread = annual_spread[(annual_spread.index.to_period("Y") < current_year) & (annual_spread.index.year >= 2019)]
annual_spread.index = annual_spread.index.year

slope, intercept = np.polyfit(annual_spread.index.values, annual_spread.values, 1)
trend = slope * annual_spread.index.values + intercept

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(annual_spread.index, annual_spread.values, color=SPREAD_COLOR, width=0.6)
ax.plot(annual_spread.index, trend, color="#4a3aa7", linewidth=2, linestyle="--", label=f"Linear trend ({slope:+.1f} EUR/MWh/year)")
ax.set_ylabel("EUR/MWh")
ax.set_xlabel("Year")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper left", frameon=False)
fig.suptitle("Average daily min-max spread, annual (2019-onward, full years only)")
fig.tight_layout()
plt.show()

print(f"Linear trend: {slope:+.1f} EUR/MWh per year")
print(f"{annual_spread.index.min()}: {annual_spread.iloc[0]:.0f} EUR/MWh -> {annual_spread.index.max()}: {annual_spread.iloc[-1]:.0f} EUR/MWh")

# %% [markdown]
# The average daily spread roughly quadrupled from 2019 to 2025 (2022's gas
# crisis is a one-year spike on top of that trend, not the source of it --
# 2023-2025 stay well above the pre-crisis level). More spread, on its own,
# just means "more arbitrage opportunity" -- it says nothing yet about
# *shape*, which is what the next section looks at.

# %% [markdown]
# ## Average daily shape, by season and year
#
# One average hourly curve per season-year: for every hour of the day (0-23),
# the mean price across all days of that season in that year. Summer =
# June-August. Winter = December-February, labeled by the January it
# contains (so "winter 2019" = Dec 2018 + Jan-Feb 2019) -- the standard
# meteorological-winter convention, needed because December and
# January/February would otherwise land in different calendar years. Only
# seasons that have fully completed are kept, same exclusion logic as the
# monthly/annual views above.

# %%
hourly = price.copy()
hourly["hour"] = hourly.index.hour
hourly["month"] = hourly.index.month
hourly["summer_year"] = hourly.index.year
hourly["winter_year"] = np.where(hourly["month"] == 12, hourly.index.year + 1, hourly.index.year)

SUMMER_MONTHS = [6, 7, 8]
WINTER_MONTHS = [12, 1, 2]

summer_curves = (
    hourly[hourly["month"].isin(SUMMER_MONTHS)]
    .groupby(["summer_year", "hour"])["price_de_lu"].mean()
    .unstack("hour")
)
summer_curves = summer_curves[[pd.Period(f"{y}-08") < current_month for y in summer_curves.index]]

winter_curves = (
    hourly[hourly["month"].isin(WINTER_MONTHS)]
    .groupby(["winter_year", "hour"])["price_de_lu"].mean()
    .unstack("hour")
)
winter_curves = winter_curves[[pd.Period(f"{y}-02") < current_month for y in winter_curves.index]]

print(f"Summer seasons: {list(summer_curves.index)}")
print(f"Winter seasons: {list(winter_curves.index)}")


# %%
def plot_seasonal_curves(ax, curves, cmap_name, title):
    years = curves.index.to_numpy()
    cmap = mpl.colormaps[cmap_name]
    positions = np.linspace(0.35, 0.95, len(years))
    for year, pos in zip(years, positions):
        ax.plot(curves.columns, curves.loc[year], color=cmap(pos), linewidth=2)
    ax.set_xlabel("Hour of day")
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title)
    norm = mcolors.Normalize(vmin=years.min(), vmax=years.max())
    sm = cm.ScalarMappable(cmap=mcolors.LinearSegmentedColormap.from_list("", [cmap(p) for p in positions]), norm=norm)
    cbar = plt.colorbar(sm, ax=ax, ticks=years)
    cbar.ax.set_yticklabels([str(y) for y in years])


fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
plot_seasonal_curves(axes[0], summer_curves, SUMMER_CMAP, "Summer (Jun-Aug)")
plot_seasonal_curves(axes[1], winter_curves, WINTER_CMAP, "Winter (Dec-Feb)")
axes[0].set_ylabel("EUR/MWh (average price for that hour of day)")
fig.suptitle("Average daily price shape, by year")
fig.tight_layout()
plt.show()

# %% [markdown]
# The two panels tell different stories, exactly as expected. **Summer**
# starts (2019-2021) as one broad daytime hump with only a shallow midday
# dip; by 2023-2026 that dip has turned into a deep trough with two clearly
# separate peaks either side of it -- a small morning one and a larger
# evening one, opening up right around the hours PV output is highest.
# **Winter** stays essentially one-humped throughout, 2019 through 2026 --
# with little PV output to suppress midday demand-driven prices in the
# first place, there's no mechanism here to split the peak. That contrast
# is itself the check on the hypothesis: the shape change tracks the
# season where PV actually produces, not the calendar in general.

# %% [markdown]
# ## Same shapes, with the price level taken out
#
# The chart above mixes two things at once: the overall price *level*,
# which varies enormously year to year (2022's crisis alone dwarfs the
# 2019-2021 lines), and the intraday *shape*, which is what this page is
# actually about. Subtracting each year's own average price from its curve
# removes the level and leaves only the shape -- every line is centered on
# zero, so a 2019 curve and a 2026 curve sit on directly comparable footing
# regardless of how far apart their price levels were.

# %%
summer_demeaned = summer_curves.sub(summer_curves.mean(axis=1), axis=0)
winter_demeaned = winter_curves.sub(winter_curves.mean(axis=1), axis=0)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
plot_seasonal_curves(axes[0], summer_demeaned, SUMMER_CMAP, "Summer (Jun-Aug)")
plot_seasonal_curves(axes[1], winter_demeaned, WINTER_CMAP, "Winter (Dec-Feb)")
for ax in axes:
    ax.axhline(0, color="#9a9990", linewidth=1)
axes[0].set_ylabel("Deviation from that year's average price (EUR/MWh)")
fig.suptitle("Average daily price shape, demeaned per year")
fig.tight_layout()
plt.show()

# %% [markdown]
# With the level stripped out, the summer amplitude growth is even more
# striking: the peak-to-trough swing around each year's own average goes
# from ~20 EUR/MWh in 2019-2020 to ~130-175 EUR/MWh in 2024-2026 (2022, the
# crisis year, is its own outlier at ~210, driven by volatility rather than
# a structurally deeper trough). Winter's swing grows too, but far more
# mildly -- roughly 25-30 EUR/MWh in 2019-2021 to 45-80 EUR/MWh since,
# without a clean upward trend, and largely tracking the 2022/2023 crisis
# rather than a steady structural shift. Demeaning doesn't change the
# conclusion, it sharpens it: summer's *relative* daily swing has grown
# several times over, winter's has barely moved.

# %% [markdown]
# ## Watching the shape emerge, year by year
#
# Same demeaned curves, animated: one year added per frame, prior years left
# in as a light grey trace behind it, so the progression is visible frame by
# frame rather than needing to be read out of eight overlaid lines at once.
# Both panels share one fixed y-axis across every frame (set from the
# global min/max across all years) so the growing amplitude is a genuine
# visual change, not an axis rescale.

# %%
YLIM = 130  # covers the global demeaned min/max across both panels with a small margin

frames = []
for i, year in enumerate(summer_curves.index):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    for j in range(i):
        prior_year = summer_curves.index[j]
        axes[0].plot(summer_demeaned.columns, summer_demeaned.loc[prior_year], color="#c9c7c0", linewidth=1.3, alpha=0.7, zorder=1)
        axes[1].plot(winter_demeaned.columns, winter_demeaned.loc[prior_year], color="#c9c7c0", linewidth=1.3, alpha=0.7, zorder=1)

    axes[0].plot(summer_demeaned.columns, summer_demeaned.loc[year], color=mpl.colormaps[SUMMER_CMAP](0.75), linewidth=2.5, zorder=2)
    axes[1].plot(winter_demeaned.columns, winter_demeaned.loc[year], color=mpl.colormaps[WINTER_CMAP](0.75), linewidth=2.5, zorder=2)

    for ax, title in zip(axes, ["Summer (Jun-Aug)", "Winter (Dec-Feb)"]):
        ax.axhline(0, color="#9a9990", linewidth=1)
        ax.set_xlabel("Hour of day")
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.set_ylim(-YLIM, YLIM)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(title)
    axes[0].set_ylabel("Deviation from that year's average price (EUR/MWh)")
    fig.suptitle(f"Average daily price shape, demeaned -- {year}")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    frames.append(Image.open(buf).convert("RGB"))

# Saved next to where the executed notebook lands -- anchored via
# INSIGHTS_ROOT rather than a relative path, same reasoning as
# `01_mastr_capacity_de`'s capacity_map_animation.gif (jupytext executes
# with cwd relative to the source file in pages/, not the repo root).
gif_path = INSIGHTS_ROOT / "book" / "notebooks" / "price_shape_animation.gif"
frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=700, loop=0)
print(f"Saved {len(frames)}-frame animation -> {gif_path}")

# %% [markdown]
# ```{figure} price_shape_animation.gif
# :name: fig-price-shape-animation
# Average daily price shape (demeaned), summer and winter side by side,
# animated year by year with prior years left in as a grey trace.
# ```
