"""Precompute LLM summaries so /world-summary and /country-detail don't
call the LLM on every request. Run whenever forecast data refreshes.
Resumable: skips any scope already present in the existing cache file.

Usage:
    python model/generate_llm_summaries.py
"""

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from package_folder.climate import get_global_movers, get_country_detail, all_predictions
from package_folder.llm_integration import summarize_world_map, summarize_country_detail

output_path = ROOT / "data" / "outputs" / "llm_summaries_cache.csv"

if output_path.exists():
    existing = pd.read_csv(output_path)
    rows = existing.to_dict("records")
    done_scopes = set(existing["scope"])
    print(f"Resuming: {len(done_scopes)} scopes already cached.")
else:
    rows = []
    done_scopes = set()

if "world" not in done_scopes:
    rows.append({"scope": "world", "summary": summarize_world_map(get_global_movers(top_n=5))})
    done_scopes.add("world")

countries = sorted({r["country"] for r in all_predictions()})
for i, country in enumerate(countries):
    if country in done_scopes:
        continue
    if i % 20 == 0:
        print(f"Generating summary {i + 1}/{len(countries)}: {country}")
    try:
        detail = get_country_detail(country)
        rows.append({"scope": country, "summary": summarize_country_detail(detail)})
    except Exception as error:
        print(f"Skipping {country}: {error}")

    # Save incrementally every 20 countries, so a crash doesn't lose everything
    if i % 20 == 0:
        pd.DataFrame(rows).to_csv(output_path, index=False)

pd.DataFrame(rows).to_csv(output_path, index=False)
print(f"Saved {len(rows)} cached summaries to {output_path}")
