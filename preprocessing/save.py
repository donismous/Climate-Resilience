

def save_processed(df):
    """
    Save the processed DataFrame to a Parquet file.
    """
    df.to_parquet("processed_data.parquet", index=False)
