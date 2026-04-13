import numpy as np
import pickle
from collections import Counter
from tensorflow.keras.models import load_model

ART_PATH   = "artifacts/"
SHARED_DIR = "../../artifacts/shared/"
N          = 500

model          = load_model(ART_PATH + "la_model.keras")
target_scaler  = pickle.load(open(ART_PATH + "target_scaler.pkl", "rb"))
pitcher_le     = pickle.load(open(SHARED_DIR + "pitcher_le.pkl",  "rb"))
batter_le      = pickle.load(open(SHARED_DIR + "batter_le.pkl",   "rb"))
bat_speed_lookup   = pickle.load(open(SHARED_DIR + "bat_speed_lookup.pkl",   "rb"))
bat_speed_pop_mean = pickle.load(open(SHARED_DIR + "bat_speed_pop_mean.pkl", "rb"))
ctx_scaler  = pickle.load(open(ART_PATH + "ctx_scaler.pkl",  "rb"))
loc_scaler  = pickle.load(open(ART_PATH + "loc_scaler.pkl",  "rb"))

ALL_PITCH_TYPES = ["FF", "SI", "FC", "SL", "CU", "CH", "FS", "KC", "ST", "SV", "CS", "FO", "KN", "EP"]

rng = np.random.default_rng(42)

# Pitch type one-hot
pt_idx = rng.integers(0, len(ALL_PITCH_TYPES), N)
PT = np.zeros((N, len(ALL_PITCH_TYPES)), dtype=np.float32)
PT[np.arange(N), pt_idx] = 1.0

# Location
plate_x = rng.uniform(-1.5, 1.5, N).astype(np.float32)
plate_z  = rng.uniform(1.0,  4.5, N).astype(np.float32)
dist     = np.sqrt(plate_x**2 + (plate_z - 2.5)**2)
is_str   = ((np.abs(plate_x) <= 0.83) & (plate_z >= 1.5) & (plate_z <= 3.5)).astype(np.float32)
LOC      = np.column_stack([plate_x, plate_z, dist, is_str]).astype(np.float32)

# Context: balls, strikes, outs, inning, score_diff, bat_speed_val,
#          on_1b, on_2b, on_3b, has_bat_speed, stand_L, stand_R, p_throws_L, p_throws_R
bat_speed_vals = rng.uniform(60, 85, N).astype(np.float32)  # realistic bat speed range
stand_L = rng.integers(0, 2, N).astype(np.float32)
p_throws_L = rng.integers(0, 2, N).astype(np.float32)

CTX = np.column_stack([
    rng.integers(0, 4, N),           # balls
    rng.integers(0, 3, N),           # strikes
    rng.integers(0, 3, N),           # outs
    rng.integers(1, 10, N),          # inning
    rng.uniform(-5, 5, N),           # score_diff
    bat_speed_vals,                   # bat_speed_val
    rng.integers(0, 2, N),           # on_1b
    rng.integers(0, 2, N),           # on_2b
    rng.integers(0, 2, N),           # on_3b
    np.ones(N),                       # has_bat_speed
    stand_L,                          # stand_L
    1 - stand_L,                      # stand_R
    p_throws_L,                       # p_throws_L
    1 - p_throws_L,                   # p_throws_R
]).astype(np.float32)

# Apply scalers
LOC_s = loc_scaler.transform(LOC)
CTX_s = CTX.copy()
CTX_s[:, :6] = ctx_scaler.transform(CTX[:, :6])

# Pitcher/batter IDs
P = rng.integers(0, len(pitcher_le.classes_), N).astype(np.int32)
B = rng.integers(0, len(batter_le.classes_),  N).astype(np.int32)

# Predict
preds_scaled = model.predict([PT, LOC_s, CTX_s, P, B], batch_size=256, verbose=0)
preds_real   = target_scaler.inverse_transform(preds_scaled).flatten()

print(f"LA — min: {preds_real.min():.1f}  max: {preds_real.max():.1f}  "
      f"mean: {preds_real.mean():.1f}  std: {preds_real.std():.1f}")

def la_bucket(la):
    if la < 10:   return "Groundball"
    elif la < 25: return "Line Drive"
    elif la < 50: return "Fly Ball"
    else:         return "Popup"

buckets = Counter(la_bucket(la) for la in preds_real)
total   = sum(buckets.values())
print("\nLA bucket distribution:")
for name in ["Groundball", "Line Drive", "Fly Ball", "Popup"]:
    count = buckets.get(name, 0)
    print(f"  {name:12s}: {count:4d}  ({100*count/total:.1f}%)")
