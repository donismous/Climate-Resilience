""" Preprocessing pipeline for ND-GAIN data. """


from preprocessing.filter import filter_indicators
from preprocessing.clean import clean_data
from preprocessing.validate import validate
from preprocessing.prepare import prepare_for_model


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

    df = prepare_for_model(df)
    print(f"Prepared dataset: {df.shape}")

    return df
