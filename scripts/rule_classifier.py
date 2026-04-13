import pandas as pd
import numpy as np

CSV_DIR = "../csv data"
YEARS = [2021, 2022, 2023, 2024, 2025]

BALLS_IN_PLAY = {"hit_into_play", "hit_into_play_no_out", "hit_into_play_score"}

EV_BUCKET_EDGES = [0, 70, 85, 100, 200]
EV_BUCKET_NAMES = ["Soft", "Medium", "Hard", "Barrel"]

def ev_bucket(ev):
    for i, (lo, hi) in enumerate(zip(EV_BUCKET_EDGES, EV_BUCKET_EDGES[1:])):
        if lo <= ev < hi:
            return EV_BUCKET_NAMES[i]
    return "Barrel"

def la_cat(la):
    if la < 10:  return "groundball"
    elif la < 25: return "line_drive"
    elif la < 50: return "fly_ball"
    else:         return "popup"

dfs = []
for y in YEARS:
    dfs.append(pd.read_csv(f"{CSV_DIR}/statcast_full_{y}.csv"))
df = pd.concat(dfs, ignore_index=True)

df = df[df["description"].isin(BALLS_IN_PLAY)].copy()
df = df.dropna(subset=["launch_speed", "launch_angle", "events"])

df["ev_bucket"] = df["launch_speed"].map(ev_bucket)
df["la_cat"]    = df["launch_angle"].map(la_cat)

HIT_EVENTS = {"single", "double", "triple", "home_run"}

for bucket in EV_BUCKET_NAMES:
    for cat in ["groundball", "line_drive", "fly_ball", "popup"]:
        sub = df[(df["ev_bucket"] == bucket) & (df["la_cat"] == cat)]
        if len(sub) < 50:
            continue
        counts = sub["events"].value_counts(normalize=True)
        print(f"\n{bucket} + {cat}  (n={len(sub):,})")
        for evt, pct in counts.head(6).items():
            print(f"  {evt:<20} {pct*100:.1f}%")
