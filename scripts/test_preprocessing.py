
from data.inputs.fetch.fetch_ndgain import fetch_ndgain
from data.preprocessing.pipeline import preprocess

df = fetch_ndgain()

print("Raw shape:", df.shape)

processed = preprocess(df)

print("Processed shape:", processed.shape)

print(processed.head())
print(processed.info())


df.to_parquet("test.parquet")
