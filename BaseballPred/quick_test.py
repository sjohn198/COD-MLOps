from pybaseball import statcast, statcast_single_game
import pandas as pd

df_rd = statcast(start_dt="2025-04-22", end_dt="2025-04-23", team="SEA")

print(list(df_rd.columns))