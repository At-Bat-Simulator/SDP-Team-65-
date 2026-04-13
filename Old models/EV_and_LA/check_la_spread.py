import numpy as np
import pickle
from collections import Counter
from tensorflow.keras.models import load_model

ART_PATH   = "artifacts/"
SHARED_DIR = "../../artifacts/shared/"
N          = 500
SEQ_LEN    = 3

model          = load_model(ART_PATH + "ev_model.keras")
target_scaler  = pickle.load(open(ART_PATH + "target_scaler.pkl", "rb"))
pitcher_le     = pickle.load(open(SHARED_DIR + "pitcher_le.pkl", "rb"))
batter_le      = pickle.load(open(SHARED_DIR + "batter_le.pkl",  "rb"))

ALL_PITCH_TYPES = ["FF", "SI", "FC", "SL", "CU", "CH", "FS", "KC", "ST", "SV", "CS", "FO", "KN", "EP"]

rng = np.random.default_rng(42)

# Sequence: 17 features — balls, strikes, outs, inning, bat_score, fld_score, score_diff,
# on_1b, on_2b, on_3b, stand_L, stand_R, p_throws_L, p_throws_R, prev_plate_x, prev_plate_z, (padding)
SEQ = np.zeros((N, SEQ_LEN, 17), dtype=np.float32)
SEQ[:, :, 0] = rng.integers(0, 4, (N, SEQ_LEN))       # balls
SEQ[:, :, 1] = rng.integers(0, 3, (N, SEQ_LEN))       # strikes
SEQ[:, :, 2] = rng.integers(0, 3, (N, SEQ_LEN))       # outs
SEQ[:, :, 3] = rng.integers(1, 10, (N, SEQ_LEN))      # inning
SEQ[:, :, 4] = rng.integers(0, 10, (N, SEQ_LEN))      # bat_score
SEQ[:, :, 5] = rng.integers(0, 10, (N, SEQ_LEN))      # fld_score
SEQ[:, :, 6] = rng.uniform(-5, 5, (N, SEQ_LEN))       # score_diff
SEQ[:, :, 7] = rng.integers(0, 2, (N, SEQ_LEN))       # on_1b
SEQ[:, :, 8] = rng.integers(0, 2, (N, SEQ_LEN))       # on_2b
SEQ[:, :, 9] = rng.integers(0, 2, (N, SEQ_LEN))       # on_3b
SEQ[:, :, 10] = rng.integers(0, 2, (N, SEQ_LEN))      # stand_L
SEQ[:, :, 11] = 1 - SEQ[:, :, 10]                     # stand_R
SEQ[:, :, 12] = rng.integers(0, 2, (N, SEQ_LEN))      # p_throws_L
SEQ[:, :, 13] = 1 - SEQ[:, :, 12]                     # p_throws_R
SEQ[:, :, 14] = rng.uniform(-1.5, 1.5, (N, SEQ_LEN))  # prev_plate_x
SEQ[:, :, 15] = rng.uniform(1.0, 4.0, (N, SEQ_LEN))   # prev_plate_z

# Pitcher/batter IDs — random valid IDs
P = rng.integers(0, len(pitcher_le.classes_), N).astype(np.int32)
B = rng.integers(0, len(batter_le.classes_),  N).astype(np.int32)

# Pitch type one-hot
pt_idx = rng.integers(0, len(ALL_PITCH_TYPES), N)
PT = np.zeros((N, len(ALL_PITCH_TYPES)), dtype=np.float32)
PT[np.arange(N), pt_idx] = 1.0

# Location — random plausible plate_x, plate_z
plate_x = rng.uniform(-1.5, 1.5, N).astype(np.float32)
plate_z  = rng.uniform(1.0,  4.5, N).astype(np.float32)
dist     = np.sqrt(plate_x**2 + (plate_z - 2.5)**2)
is_str   = ((np.abs(plate_x) <= 0.83) & (plate_z >= 1.5) & (plate_z <= 3.5)).astype(np.float32)
LOC      = np.column_stack([plate_x, plate_z, dist, is_str]).astype(np.float32)

# Predict
preds_scaled = model.predict([SEQ, P, B, PT, LOC], batch_size=256, verbose=0)
preds_real   = target_scaler.inverse_transform(preds_scaled)

ev_preds = preds_real[:, 0]
la_preds = preds_real[:, 1]

print(f"EV — min: {ev_preds.min():.1f}  max: {ev_preds.max():.1f}  mean: {ev_preds.mean():.1f}  std: {ev_preds.std():.1f}")
print(f"LA — min: {la_preds.min():.1f}  max: {la_preds.max():.1f}  mean: {la_preds.mean():.1f}  std: {la_preds.std():.1f}")

def la_bucket(la):
    if la < 10:  return "Groundball"
    elif la < 25: return "Line Drive"
    elif la < 50: return "Fly Ball"
    else:         return "Popup"

buckets = Counter(la_bucket(la) for la in la_preds)
total   = sum(buckets.values())
print("\nLA bucket distribution:")
for name in ["Groundball", "Line Drive", "Fly Ball", "Popup"]:
    count = buckets.get(name, 0)
    print(f"  {name:12s}: {count:4d}  ({100*count/total:.1f}%)")
