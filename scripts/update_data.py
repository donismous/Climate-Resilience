

from data.inputs.fetch.fetch_ndgain import fetch_ndgain
from data.preprocessing.pipeline import preprocess
from data.preprocessing.save import save_processed

raw_df = fetch_ndgain()
processed_df = preprocess(raw_df)
save_processed(processed_df)
