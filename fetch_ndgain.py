"""Fetch ND-GAIN climate resilience data from the World Bank Data360 API.

The ND-GAIN (Notre Dame Global Adaptation Initiative) dataset scores countries
on their vulnerability to climate change and their readiness to adapt. The
World Bank republishes it through its Data360 API, which this script queries.

API docs: https://data360.worldbank.org/en/api?datasetid=UND_NDGAIN

Usage:
    Run as a script to download the full dataset, preview it, and save it
    to ``ndgain_raw.csv``:

        python fetch_ndgain.py

    Or import and filter from another module / notebook:

        from fetch_ndgain import fetch_ndgain
        df = fetch_ndgain(indicator="UND_NDGAIN_VULNER", country="FRA")
"""

import requests
import pandas as pd

# Root endpoint for data queries on the Data360 API.
BASE_URL = "https://data360api.worldbank.org/data360/data"

# Identifier of the ND-GAIN dataset within Data360.
DATASET_ID = "UND_NDGAIN"

# The API caps each response at 1000 rows, so larger downloads must be
# paginated with the `top` (page size) and `skip` (offset) parameters.
PAGE_SIZE = 1000


def fetch_ndgain(indicator: str = None, country: str = None) -> pd.DataFrame:
    """Download ND-GAIN data as a pandas DataFrame.

    Pages through the Data360 API (1000 rows per request) until every row
    matching the filters has been retrieved, printing progress along the way.

    Args:
        indicator: Optional indicator code to filter on, e.g.
            "UND_NDGAIN_VULNER" (vulnerability score). If omitted, all
            indicators in the dataset are returned.
        country: Optional ISO3 country code to filter on, e.g. "FRA".
            If omitted, all countries are returned.

    Returns:
        A DataFrame with one row per (country, indicator, year) observation,
        using the API's raw column names (REF_AREA, INDICATOR, TIME_PERIOD,
        OBS_VALUE, ...). Empty if no rows match the filters.

    Raises:
        requests.HTTPError: If the API returns an error status code.
    """
    params = {"DATABASE_ID": DATASET_ID, "top": PAGE_SIZE, "skip": 0}
    if indicator:
        params["INDICATOR"] = indicator
    if country:
        params["REF_AREA"] = country

    rows = []
    while True:
        response = requests.get(BASE_URL, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()

        # Each response holds one page of rows under "value" and the total
        # row count for the query under "count".
        page = data.get("value", [])
        rows.extend(page)
        print(f"Fetched {len(rows)} / {data['count']} rows")

        # Stop once we have every row, or if the API returns an empty page
        # (guards against an infinite loop if "count" is ever inconsistent).
        if len(rows) >= data["count"] or not page:
            break
        params["skip"] += PAGE_SIZE

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Download the full dataset, show a preview, and save a local raw copy.
    df = fetch_ndgain()
    print(df.head())
    df.to_csv("ndgain_raw.csv", index=False)
    print(f"Saved {len(df)} rows to ndgain_raw.csv")
