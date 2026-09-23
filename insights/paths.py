"""Paths into energy-data-hub's data/ -- this repo owns no data of its own.

Same convention documented in energy-data-hub/README.md's "How a book repo
consumes hub data": absolute path into the sibling hub checkout, fail loudly
if it's not there (this machine doesn't have the hub cloned/materialized)
rather than silently producing an empty page.
"""

from pathlib import Path

INSIGHTS_ROOT = Path(__file__).resolve().parent.parent
HUB_DATA = Path.home() / "research" / "energy-data-hub" / "data"


def hub_file(*parts: str) -> Path:
    """Path to a hub data file, e.g. `hub_file("capacity", "capacity_events.parquet")`."""
    path = HUB_DATA.joinpath(*parts)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- is energy-data-hub cloned as a sibling repo, "
            f"with its `{'/'.join(parts[:-1]) or '.'}` asset(s) materialized?"
        )
    return path
