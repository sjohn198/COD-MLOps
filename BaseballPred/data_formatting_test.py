from pybaseball import statcast, statcast_single_game
import pandas as pd


#This code formats the statcast data into trainable rows with labels to allow for win prediction

df_rd = statcast(start_dt="2025-04-22", end_dt="2025-04-23", team="SEA")

features = ["game_date", "pitcher", "batter", "home_team", "away_team", "balls", "game_pk", "at_bat_number", 
            "pitch_number", "strikes", "on_3b", "on_2b", "on_1b", "outs_when_up", "inning", "home_score", "away_score"]
cleaned_df = df_rd[features].sort_values(["game_pk", "at_bat_number", "pitch_number"])
print(cleaned_df)

final_score_df = cleaned_df.drop_duplicates(subset=['game_pk'], keep='last')
print(final_score_df)

final_score_df["winner"] = (final_score_df["home_score"] - final_score_df["away_score"]) / abs(final_score_df["home_score"] - final_score_df["away_score"])
final_score_df = final_score_df[["game_pk", "winner"]]
print(final_score_df)

cleaned_df = cleaned_df.merge(final_score_df, on="game_pk", how="inner")
print(cleaned_df)

