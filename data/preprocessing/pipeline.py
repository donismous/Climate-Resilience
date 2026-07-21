""" Preprocessing pipeline for ND-GAIN data. """



import pandas as pd


from data.preprocessing.filter import filter_indicators
from data.preprocessing.clean import clean_data
from data.preprocessing.validate import validate
from data.preprocessing.prepare import prepare_for_model
from data.preprocessing.save import save_processed


def preprocess(df):
    """
    Run the complete preprocessing pipeline.
    """

    print(f"Downloaded {len(df):,} rows")

    df = filter_indicators(df)
    print(f"Keeping {len(df):,} ND-GAIN indicator observations")

    df = clean_data(df)
    print(f"Cleaned data: {df.shape}")

    validate(df)
    print(f"Validated dataset: {df.shape}")

    df = prepare_for_model(df)
    print(f"Prepared dataset: {df.shape}")

    df = save_processed(df)
    print(f"Saved processed data: data/data_preprocessed/processed_data.csv")

    return df


# Load and run
if __name__ == "__main__":

# Load your raw data
    df = pd.read_csv("data/inputs/fetch/ndgain_raw.csv")
    processed_df = preprocess(df)
