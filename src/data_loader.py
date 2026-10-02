import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "dubai_residential_data_2026.csv"


def load_data():
    """Load the Dubai property transaction dataset."""
    
    df = pd.read_csv(DATA_PATH)
    
    print(f"Dataset loaded successfully.")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")
    
    return df


if __name__ == "__main__":
    df = load_data()
    
    print("\nFirst 5 rows:")
    print(df.head())