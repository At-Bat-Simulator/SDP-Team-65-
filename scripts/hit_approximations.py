import pandas as pd
import numpy as np

df = pd.read_csv("csv data/statcast_full_2021.csv")
df = pd.read_csv("csv data/statcast_full_2022.csv")
df = pd.read_csv("csv data/statcast_full_2023.csv")
df = pd.read_csv("csv data/statcast_full_2024.csv")
df = pd.read_csv("csv data/statcast_full_2025.csv")

# Filter to fair balls with a clear outcome
fair = df[df["events"].isin(["single","double","home_run","field_out","grounded_into_double_play","force_out","sac_fly"])].copy()
fair = fair.dropna(subset=["launch_speed","launch_angle"])

# Group by outcome and show EV/LA percentiles
for event, group in fair.groupby("events"):
    print(f"\n{event} (n={len(group)})")
    print(f"  EV:  mean={group['launch_speed'].mean():.1f}  p25={group['launch_speed'].quantile(.25):.1f}  p75={group['launch_speed'].quantile(.75):.1f}")
    print(f"  LA:  mean={group['launch_angle'].mean():.1f}  p25={group['launch_angle'].quantile(.25):.1f}  p75={group['launch_angle'].quantile(.75):.1f}")
