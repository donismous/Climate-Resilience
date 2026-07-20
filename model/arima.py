

import data.inputs.fetch.fetch_ndgain as fetch_ndgain
from data.preprocessing.pipeline import preprocess


raw_df = fetch_ndgain()
prepared_df = preprocess(raw_df)




Below code is to manage missing values in the time series data for ARIMA modeling.


def prepare_country_series(df, country, indicator):

    series = (
        df.loc[df["Country"] == country,
               ["Year", indicator]]
          .set_index("Year")
          .sort_index()
    )

    return series


for country in countries:

    series = prepare_country_series(
        prepared_df,
        country,
        "Capacity"
    )

    if len(series.dropna()) < 20:
        continue

#     train_arima(series)
