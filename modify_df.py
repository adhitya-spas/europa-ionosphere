import pandas as pd

# All angles used in this file should be interpreted as true incidence angle (from vertical, 0 = nadir, 90 = horizontal)

def save_mission_df(df: pd.DataFrame, path: str = "new_mission_df.pkl") -> None:
    """
    Save the mission DataFrame to disk in pickle format.
    This preserves the Python lists in the 'Altitude' and 'Ne' columns exactly.
    """
    df.to_pickle(path)
    print(f"Saved DataFrame ({len(df)} rows) to {path!r}")

def load_mission_df(path: str = "new_mission_df.pkl") -> pd.DataFrame:
    """
    Load the mission DataFrame back from disk.
    Returns a DataFrame with exactly the same structure you saved.
    """
    df = pd.read_pickle(path)
    print(f"Loaded DataFrame ({len(df)} rows) from {path!r}")
    return df