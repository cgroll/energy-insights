from dagster import Definitions

from insights_dagster.assets import page_assets

defs = Definitions(assets=page_assets)
