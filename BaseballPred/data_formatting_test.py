from pybaseball import statcast, statcast_single_game
import pandas as pd
import numpy as np


#This code formats the statcast data into trainable rows with labels to allow for win prediction

df_rd = statcast(start_dt="2025-04-22", end_dt="2025-04-23", team="SEA")
print(df_rd["inning_topbot"].value_counts())

features = ["balls", "game_pk", "at_bat_number", "inning_topbot",
            "pitch_number", "strikes", "on_3b", "on_2b", "on_1b", "outs_when_up", "inning", "home_score", "away_score"]
cleaned_df = df_rd[features].sort_values(["game_pk", "at_bat_number", "pitch_number"])
cleaned_df["on_3b"] = cleaned_df["on_3b"].notna().astype(int)
cleaned_df["on_2b"] = cleaned_df["on_2b"].notna().astype(int)
cleaned_df["on_1b"] = cleaned_df["on_1b"].notna().astype(int)
cleaned_df["inning_topbot"] = (cleaned_df["inning_topbot"] == "Bot").astype(int)
print(cleaned_df["inning_topbot"].value_counts())

final_score_df = cleaned_df.drop_duplicates(subset=['game_pk'], keep='last').copy()
#print(final_score_df)

final_score_df["winner"] = np.sign(final_score_df["home_score"] - final_score_df["away_score"] + 1e-6)
final_score_df = final_score_df[["game_pk", "winner"]]
#print(final_score_df)

cleaned_df = cleaned_df.merge(final_score_df, on="game_pk", how="inner")
#print(cleaned_df)

