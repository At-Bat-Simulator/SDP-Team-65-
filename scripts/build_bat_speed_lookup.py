import pandas as pd
import pickle

CSV_DIR    = "../csv data"
SHARED_DIR = "../artifacts/shared/"
YEARS      = [2024, 2025]

dfs = []
for y in YEARS:
    path = f"{CSV_DIR}/statcast_full_{y}.csv"
    print("Loading:", path)
    dfs.append(pd.read_csv(path, low_memory=False))

df = pd.concat(dfs, ignore_index=True)

df = df.dropna(subset=["batter", "bat_speed"])
df["batter"] = df["batter"].astype(int)

lookup = df.groupby("batter")["bat_speed"].mean().to_dict()
pop_mean = df["bat_speed"].mean()

print(f"Batters with bat speed data: {len(lookup):,}")
print(f"Population mean bat speed: {pop_mean:.2f} mph")
print(f"Min mean: {min(lookup.values()):.1f}  Max mean: {max(lookup.values()):.1f}")

pickle.dump(lookup,   open(SHARED_DIR + "bat_speed_lookup.pkl",   "wb"))
pickle.dump(pop_mean, open(SHARED_DIR + "bat_speed_pop_mean.pkl", "wb"))

print("\n✓ Saved bat_speed_lookup.pkl and bat_speed_pop_mean.pkl")
