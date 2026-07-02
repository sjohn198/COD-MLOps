import os
import pandas as pd
from pybaseball import statcast, cache
from datetime import timedelta
import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

cache.enable()

def format_data(df):
    features = ["balls", "game_pk", "at_bat_number", "inning_topbot",
            "pitch_number", "strikes", "on_3b", "on_2b", "on_1b", "outs_when_up", "inning", "home_score", "away_score"]
    
    cleaned_df = df[features].sort_values(["game_pk", "at_bat_number", "pitch_number"])
    cleaned_df["on_3b"] = cleaned_df["on_3b"].notna().astype(int)
    cleaned_df["on_2b"] = cleaned_df["on_2b"].notna().astype(int)
    cleaned_df["on_1b"] = cleaned_df["on_1b"].notna().astype(int)
    cleaned_df["inning_topbot"] = (cleaned_df["inning_topbot"] == "Bot").astype(int)

    final_score_df = cleaned_df.drop_duplicates(subset=['game_pk'], keep='last').copy()

    # Calculate winner: -1 (Away), 0 (Tie), 1 (Home)
    final_score_df["winner"] = np.sign(final_score_df["home_score"] - final_score_df["away_score"])
    
    # Drop tie games to fit the binary classification (out_features=2)
    final_score_df = final_score_df[final_score_df["winner"] != 0]
    
    # Map PyTorch-compatible class labels: 0 for Away Win, 1 for Home Win
    final_score_df["winner"] = final_score_df["winner"].map({-1: 0, 1: 1})
    
    final_score_df = final_score_df[["game_pk", "winner"]]

    cleaned_df = cleaned_df.merge(final_score_df, on="game_pk", how="inner")

    cleaned_df["run_diff"] = cleaned_df["home_score"] - cleaned_df["away_score"]
    
    return cleaned_df[["balls", "inning_topbot", "strikes", "on_3b", "on_2b", "on_1b", "outs_when_up", "inning", "run_diff", "winner"]]


def read_by_year(year, dataset_dir):
    start_date = pd.to_datetime(f"{year}-03-01")
    end_date = pd.to_datetime(f"{year}-11-30")

    year_dir = os.path.join(dataset_dir, f"year={year}")
    os.makedirs(year_dir, exist_ok=True)

    current_date = start_date
    while current_date <= end_date:
        chunk_end = current_date + timedelta(days=6)

        if chunk_end > end_date:
            chunk_end = end_date

        print(f"Fetching: {current_date.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}")

        cd_str = current_date.strftime("%Y-%m-%d")
        ce_str = chunk_end.strftime("%Y-%m-%d")

        max_retries = 5
        temp_df = pd.DataFrame()

        for i in range(max_retries):
            try:
                temp_df = statcast(start_dt=cd_str, end_dt=ce_str, verbose=False)
                break
            except Exception as e:
                if i < max_retries - 1:
                    time.sleep(15 * (i + 1))
                else:
                    print(f"Failed after {max_retries} attempts.")
        
        if not temp_df.empty:
            formatted_chunk = format_data(temp_df)
            table = pa.Table.from_pandas(formatted_chunk, preserve_index=False)
            
            filename = f"statcast_{cd_str}_to_{ce_str}.parquet"
            filepath = os.path.join(year_dir, filename)

            pq.write_table(table, filepath)

        time.sleep(3)
        current_date += timedelta(days=7)


if __name__ == "__main__":
    dataset_dir = "dataset"
    
    for i in range(2008, 2026):
        read_by_year(str(i), dataset_dir)
        print(f"--- Completed {i} ---")