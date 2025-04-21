import pandas as pd
import numpy as np

def load_time_series(path: str, columns: list[str]|None = None) -> np.ndarray:
    """
    Load time-series data from a CSV file.

    Args:
        path: Path to the CSV file.
        columns: Optional list of column names to select.

    Returns:
        A NumPy array of shape (n_samples, n_features).
    """
    # Read the CSV into a DataFrame
    df = pd.read_csv(path, index_col=0, parse_dates=True)  # uses pandas.read_csv :contentReference[oaicite:1]{index=1}
    # Select specified columns if provided
    if columns:
        df = df[columns]  # pandas indexing :contentReference[oaicite:2]{index=2}
    # Return as NumPy array
    return df.values  # leverages DataFrame.values to produce an ndarray :contentReference[oaicite:3]{index=3}
