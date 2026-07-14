"""Creating a merge of CSV files for one country (France)
to use for a sample ARIMA model"""


from pathlib import Path
import pandas as pd

DATA_DIR = Path("raw_data")

readiness = pd.read_csv(DATA_DIR / "readiness" / "readiness.csv")
economic = pd.read_csv(DATA_DIR / "readiness" / "sectors" / "economic.csv")
governance = pd.read_csv(DATA_DIR / "readiness" / "sectors" / "governance.csv")
social = pd.read_csv(DATA_DIR / "readiness" / "sectors" / "social.csv")
vulnerability = pd.read_csv(DATA_DIR / "vulnerability" / "vulnerability.csv")
capacity = pd.read_csv(DATA_DIR / "vulnerability" / "capacity.csv")
exposure = pd.read_csv(DATA_DIR / "vulnerability" / "exposure.csv")
sensitivity = pd.read_csv(DATA_DIR / "vulnerability" / "sensitivity.csv")
ecosystem = pd.read_csv(DATA_DIR / "vulnerability" / "sectors" / "ecosystems.csv")
food = pd.read_csv(DATA_DIR / "vulnerability" / "sectors" / "food.csv")
habitat = pd.read_csv(DATA_DIR / "vulnerability" / "sectors" / "habitat.csv")
health = pd.read_csv(DATA_DIR / "vulnerability" / "sectors" / "health.csv")
infrastructure = pd.read_csv(DATA_DIR / "vulnerability" / "sectors" / "infrastructure.csv")
water = pd.read_csv(DATA_DIR / "vulnerability" / "sectors" / "water.csv")

datasets = {
    "readiness": readiness,
    "economic": economic,
    "governance": governance,
    "social": social,
    "vulnerability": vulnerability,
    "capacity": capacity,
    "exposure": exposure,
    "sensitivity": sensitivity,
    "ecosystem": ecosystem,
    "food": food,
    "habitat": habitat,
    "health": health,
    "infrastructure": infrastructure,
    "water": water
}


france_all = pd.DataFrame()

for name, df in datasets.items():
    values = (
        df[df["ISO3"] == "FRA"]
        .drop(columns=["ISO3", "Name"])
        .T
    )

    values.columns = [name]
    france_all = pd.concat([france_all, values], axis=1)

france_all.index.name = "Year"

france_all.to_csv(DATA_DIR / "france_all.csv")
