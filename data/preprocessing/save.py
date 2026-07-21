"""
Save the processed DataFrame to a CSV file.
    """

def save_processed(df):

    df.to_csv("data/data_preprocessed/processed_data.csv", index=False)
