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

## Roadmap

- MaStR-vs-SMARD installed-capacity comparison.
- Behind-the-meter PV category breakdown.
- PV + storage combination analysis.
