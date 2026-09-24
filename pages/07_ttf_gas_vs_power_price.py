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
# # TTF gas vs. day-ahead power price
#
# Natural gas is the marginal fuel behind a large share of German day-ahead
# power prices: gas-fired plants are frequently the price-setting unit in the
# merit order, especially outside high-renewable-output hours. Comparing TTF
# (the European gas benchmark) against SMARD's DE-LU day-ahead auction price
# is a first, simple check on how tightly the two actually move together --
# both read live from `energy-data-hub`, so this stays current automatically.
#
# Both series happen to share the same unit (EUR/MWh), so they can be plotted
# on one shared axis directly -- no second y-axis, no rescaling needed.

# %%
import matplotlib.pyplot as plt
import pandas as pd

from insights.paths import hub_file

GAS_COLOR = "#2a78d6"    # dataviz skill default categorical slot 1 (blue)
POWER_COLOR = "#eb6834"  # dataviz skill default categorical slot 2 (orange)

gas_daily = pd.read_parquet(hub_file("gas", "ttf_gas_prices.parquet"))
power_hourly = pd.read_parquet(hub_file("smard", "price_de_lu.parquet"))

gas_daily.tail()

# %%
power_hourly.tail()

# %% [markdown]
# ## Monthly averages
#
# TTF is trading-day close prices, SMARD is hourly auction prices -- both
# resampled to a plain calendar-month mean. The current, still-incomplete
# month is dropped from both sides: including it would understate or
# overstate the month depending on how far it's progressed, the same
# reasoning `02_mastr_vs_smard_capacity` applies to the current year.

# %%
current_month = pd.Timestamp.today().to_period("M")

gas_monthly = gas_daily["close"].resample("MS").mean()
gas_monthly = gas_monthly[gas_monthly.index.to_period("M") < current_month]

power_monthly = power_hourly["price_de_lu"].resample("MS").mean()
power_monthly = power_monthly[power_monthly.index.to_period("M") < current_month]

monthly = pd.DataFrame({"ttf_gas_eur_mwh": gas_monthly, "power_de_lu_eur_mwh": power_monthly}).dropna()
monthly.tail()

# %% [markdown]
# ## Full history, side by side
#
# One shared EUR/MWh axis -- both series are directly comparable in level,
# not just in shape.

# %%
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(monthly.index, monthly["ttf_gas_eur_mwh"], color=GAS_COLOR, linewidth=2, label="TTF gas")
ax.plot(monthly.index, monthly["power_de_lu_eur_mwh"], color=POWER_COLOR, linewidth=2, label="Power (DE-LU day-ahead)")
ax.set_ylabel("EUR/MWh")
ax.set_xlabel("Month")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper left", frameon=False)
fig.suptitle("Monthly average TTF gas vs. DE-LU day-ahead power price")
fig.tight_layout()
plt.show()

# %% [markdown]
# The 2021-2022 gas crisis stands out clearly on both series -- power prices
# tracked gas upward almost one-for-one at the time, consistent with gas
# being the marginal price setter through most of that period.

# %% [markdown]
# ## Do the two move together?
#
# A scatter of the same monthly pairs, plus their Pearson correlation --
# a single overlay chart can look correlated just from both series trending
# in the same direction over a few big episodes (like 2021-2022 above); the
# correlation coefficient and the point cloud's scatter around a line are a
# more direct read on how tight the relationship actually is month to month.

# %%
correlation = monthly["ttf_gas_eur_mwh"].corr(monthly["power_de_lu_eur_mwh"])

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(monthly["ttf_gas_eur_mwh"], monthly["power_de_lu_eur_mwh"], color=GAS_COLOR, s=28, alpha=0.75, edgecolors="none")
ax.set_xlabel("TTF gas (EUR/MWh, monthly avg.)")
ax.set_ylabel("Power, DE-LU day-ahead (EUR/MWh, monthly avg.)")
ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(f"Monthly gas vs. power price (Pearson r = {correlation:.2f})")
fig.tight_layout()
plt.show()

print(f"Pearson correlation, monthly averages: {correlation:.2f}")
print(f"Months compared: {len(monthly)} ({monthly.index.min().date()} to {monthly.index.max().date()})")

# %% [markdown]
# ## Spread over time
#
# `power - gas`, in EUR/MWh -- both being in the same unit makes this
# difference directly meaningful, not just a shape comparison. A widening
# spread points at periods where something other than gas is driving power
# prices (e.g. carbon prices, scarcity premia, or renewable-output effects
# pulling the marginal price setter away from gas); a spread near zero is
# consistent with gas-fired plants setting the day-ahead price close to
# their own fuel cost.

# %%
spread = monthly["power_de_lu_eur_mwh"] - monthly["ttf_gas_eur_mwh"]

fig, ax = plt.subplots(figsize=(12, 4))
ax.axhline(0, color="#9a9990", linewidth=1)
ax.plot(spread.index, spread, color=POWER_COLOR, linewidth=2)
ax.set_ylabel("Power - gas (EUR/MWh)")
ax.set_xlabel("Month")
ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Power-minus-gas spread, monthly average")
fig.tight_layout()
plt.show()

spread.describe().round(1)
