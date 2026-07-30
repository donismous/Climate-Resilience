"""Precompute LLM summaries so /world-summary and /country-detail don't
call the LLM on every request. TEST RUN: world + 2 countries only.

Usage:
    python model/generate_llm_summaries.py
"""

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from package_folder.climate import get_global_movers, get_country_detail
from package_folder.llm_integration import summarize_world_map, summarize_country_detail

TEST_COUNTRIES = ["FRA", "SOM"]  # pick any two you want to check

rows = [{"scope": "world", "summary": summarize_world_map(get_global_movers(top_n=5))}]

for country in TEST_COUNTRIES:
    try:
        detail = get_country_detail(country)
        rows.append({"scope": country, "summary": summarize_country_detail(detail)})
    except Exception as error:
        print(f"Skipping {country}: {error}")

output_path = ROOT / "data" / "outputs" / "llm_summaries_cache.csv"
pd.DataFrame(rows).to_csv(output_path, index=False)
print(f"Saved {len(rows)} cached summaries to {output_path}")

for row in rows:
    print(f"\n--- {row['scope']} ---")
    print(row["summary"])
