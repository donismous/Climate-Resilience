"""Precompute LLM summaries and cacheable recommendations so the API
endpoints don't call the LLM on every request. Resumable: skips any
(scope, kind) already present in the existing cache file.

Usage:
    python model/generate_llm_summaries.py
"""

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from package_folder.climate import get_global_movers, get_country_detail, all_predictions
from package_folder.llm_integration import summarize_world_map, summarize_country_detail, draft_recommendations

CACHEABLE_PERSONAS = ["individual", "government/institution"]

TEST_COUNTRIES = ["FRA", "SOM"]  # set to None to run the full ~186-country batch

output_path = ROOT / "data" / "outputs" / "llm_summaries_cache.csv"

if output_path.exists():
    existing = pd.read_csv(output_path)
    rows = existing.to_dict("records")
    done = set(zip(existing["scope"], existing["kind"])) if "kind" in existing.columns else set()
    print(f"Resuming: {len(done)} (scope, kind) pairs already cached.")
else:
    rows = []
    done = set()


def add_row(scope: str, kind: str, summary: str):
    rows.append({"scope": scope, "kind": kind, "summary": summary})
    done.add((scope, kind))


def save():
    pd.DataFrame(rows).to_csv(output_path, index=False)


if ("world", "summary") not in done:
    world_summary_text = summarize_world_map(get_global_movers(top_n=5))
    add_row("world", "summary", world_summary_text)
else:
    world_summary_text = next(r["summary"] for r in rows if r["scope"] == "world" and r["kind"] == "summary")

for persona in CACHEABLE_PERSONAS:
    kind = f"recommendation_{persona}"
    if ("world", kind) not in done:
        rec = draft_recommendations(persona=persona, industry=None, dashboard_summary=world_summary_text, driver_summary="")
        add_row("world", kind, rec)

save()

countries = TEST_COUNTRIES if TEST_COUNTRIES else sorted({r["country"] for r in all_predictions()})
for i, country in enumerate(countries):
    if i % 20 == 0:
        print(f"Processing {i + 1}/{len(countries)}: {country}")
        save()

    country_summary_text = next((r["summary"] for r in rows if r["scope"] == country and r["kind"] == "summary"), None)
    if country_summary_text is None:
        try:
            detail = get_country_detail(country)
            country_summary_text = summarize_country_detail(detail)
            add_row(country, "summary", country_summary_text)
        except Exception as error:
            print(f"Skipping {country} summary: {error}")
            continue

    for persona in CACHEABLE_PERSONAS:
        kind = f"recommendation_{persona}"
        if (country, kind) not in done:
            try:
                rec = draft_recommendations(persona=persona, industry=None, dashboard_summary=country_summary_text, driver_summary="")
                add_row(country, kind, rec)
            except Exception as error:
                print(f"Skipping {country} {kind}: {error}")

save()
print(f"Saved {len(rows)} rows to {output_path}")

for row in rows:
    print(f"\n--- {row['scope']} / {row['kind']} ---")
    print(row["summary"])
