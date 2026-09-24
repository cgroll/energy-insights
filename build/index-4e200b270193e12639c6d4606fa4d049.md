---
title: Welcome
---

# Energy Insights

Explanatory pages on the German energy system — how much capacity exists,
where it sits, how well the official registries agree with each other, and
what that means for questions like "could today's fleet have avoided past
Dunkelflauten".

## About

Every page here is a real, executable [Dagster](https://dagster.io/) asset
(see `insights_dagster/`) that reads its data directly from
[`energy-data-hub`](https://github.com/cgroll/energy-data-hub) — the shared,
Dagster-native ingestion layer for MaStR, SMARD, and PECD data. Materializing
a page re-runs it against whatever's currently in the hub; materializing
`+page` first refreshes the hub assets it depends on. Publishing this site
(via MyST + GitHub Pages) is a separate, deliberate step from materializing —
a routine data refresh never silently republishes prose that hasn't been
reviewed against the new numbers.

## Pages

- **MaStR capacity** — installed wind + solar capacity: how much, where, and
  how it's grown over time.
- **MaStR vs. SMARD capacity** — a live sanity check of MaStR-derived
  installed capacity against the Bundesnetzagentur's own official figures.
- **PV categories** — behind-the-meter PV (grid-direct vs. self-consumption
  with/without storage): growth over time, regional mix, plant-size
  distribution, and a cross-check against usage sector / installation type.
- **PECD potential validation** — how far PECD-derived renewable potential
  sits from what SMARD reports Germany actually produced, and whether that
  gap looks like curtailment/self-consumption rather than a modeling error.
- **DE capacity factors: simple approximation vs. MaStR-weighted** — how far
  a MaStR-free approximation (fixed public technology-mix weights for solar,
  area-weighted PECD zones for wind) gets against the real MaStR-weighted
  series — a first step towards a capacity factor for countries with no
  MaStR-style registry.
- **Cross-country PECD comparison** — long-run mean capacity factors for
  solar, wind onshore, and wind offshore across every PECD country: bar
  charts, a solar-vs-wind complementarity scatter, and choropleth maps
  (offshore mapped per individual zone, not collapsed to one country color).

## Roadmap

- Seasonal/monthly capacity-factor profiles for individual countries beyond
  Germany (interannual variability, monthly distributions) — a follow-up to
  the cross-country comparison page above.
- PV + storage combination analysis.
