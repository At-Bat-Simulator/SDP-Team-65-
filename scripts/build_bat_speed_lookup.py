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
df["batter"] = df["batter"].astype(int)

# --- Bat speed ---
bs_df = df.dropna(subset=["batter", "bat_speed"])
bat_speed_lookup   = bs_df.groupby("batter")["bat_speed"].mean().to_dict()
bat_speed_pop_mean = bs_df["bat_speed"].mean()

print(f"\nBat speed:")
print(f"  Batters with data : {len(bat_speed_lookup):,}")
print(f"  Population mean   : {bat_speed_pop_mean:.2f} mph")
print(f"  Min / Max mean    : {min(bat_speed_lookup.values()):.1f} / {max(bat_speed_lookup.values()):.1f}")

pickle.dump(bat_speed_lookup,   open(SHARED_DIR + "bat_speed_lookup.pkl",   "wb"))
pickle.dump(bat_speed_pop_mean, open(SHARED_DIR + "bat_speed_pop_mean.pkl", "wb"))
print("  ✓ Saved bat_speed_lookup.pkl and bat_speed_pop_mean.pkl")

# --- Swing length ---
sl_df = df.dropna(subset=["batter", "swing_length"])
swing_length_lookup   = sl_df.groupby("batter")["swing_length"].mean().to_dict()
swing_length_pop_mean = sl_df["swing_length"].mean()

print(f"\nSwing length:")
print(f"  Batters with data : {len(swing_length_lookup):,}")
print(f"  Population mean   : {swing_length_pop_mean:.2f} ft")
print(f"  Min / Max mean    : {min(swing_length_lookup.values()):.2f} / {max(swing_length_lookup.values()):.2f}")

pickle.dump(swing_length_lookup,   open(SHARED_DIR + "swing_length_lookup.pkl",   "wb"))
pickle.dump(swing_length_pop_mean, open(SHARED_DIR + "swing_length_pop_mean.pkl", "wb"))
print("  ✓ Saved swing_length_lookup.pkl and swing_length_pop_mean.pkl")
