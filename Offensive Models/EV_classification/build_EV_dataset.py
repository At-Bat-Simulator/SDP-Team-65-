import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


TEST_SIZE = 0.15
RANDOM_SEED = 42

CSV_DIR = "../../csv data"
ARTIFACTS = "artifacts/"
SHARED_DIR = "../../artifacts/shared/"

YEARS = [2021, 2022, 2023, 2024, 2025]

BALLS_IN_PLAY = {
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
}

ALL_PITCH_TYPES = ["FF", "SI", "FC", "SL", "CU", "CH", "FS", "KC", "ST", "SV", "CS", "FO", "KN", "EP"]


# Exit velocity buckets
EV_BUCKET_EDGES = [0, 70, 85, 100, 200]
EV_BUCKET_NAMES = ["Soft", "Medium", "Hard", "Barrel"]
NUM_EV_BUCKETS  = len(EV_BUCKET_NAMES)  # 4


def ev_to_bucket(ev: float) -> int:
    for i, (lo, hi) in enumerate(zip(EV_BUCKET_EDGES, EV_BUCKET_EDGES[1:])):
        if lo <= ev < hi:
            return i
    return NUM_EV_BUCKETS - 1


def load_statcast():
    dfs = []
    for y in YEARS:
        path = f"{CSV_DIR}/statcast_full_{y}.csv"
        print("Loading:", path)
        dfs.append(pd.read_csv(path))
    return pd.concat(dfs, ignore_index=True)


def preprocess(df, pitcher_le, batter_le):
    needed =  [
        "balls", "strikes", "outs_when_up", "inning",
        "bat_score", "fld_score",
        "description", "pitch_type",
        "plate_x", "plate_z",
        "stand", "p_throws",
        "launch_speed", "launch_angle",
        "pitcher", "batter",
        "game_pk", "game_date", "at_bat_number", "pitch_number",
    ]

    df = df.dropna(subset=needed)
    df = df[df["pitch_type"].isin(ALL_PITCH_TYPES)].copy()

    for base in ["on_1b", "on_2b", "on_3b"]:
        df[base] = df[base].notna().astype(int)
    

    df = pd.get_dummies(df, columns=["stand", "p_throws"], drop_first=False)
    df["pitcher_id"] = pitcher_le.transform(df["pitcher"].astype(int))
    df["batter_id"]  = batter_le.transform(df["batter"].astype(int))
    df["score_diff"] = df["bat_score"] - df["fld_score"]

    bat_speed_lookup = pickle.load(open(SHARED_DIR + "bat_speed_lookup.pkl", "rb"))
    bat_speed_pop_mean = pickle.load(open(SHARED_DIR + "bat_speed_pop_mean.pkl", "rb"))


    df["bat_speed_val"] = df["batter"].astype(int).map(
        lambda x: bat_speed_lookup.get(x, bat_speed_pop_mean)
    )

    df["has_bat_speed"] = df["batter"].astype(int).map(
    lambda x: 1.0 if x in bat_speed_lookup else 0.0
    )

    return df

def build_pitch_type_onehot(pitch_type: str) -> np.ndarray:
    vec = np.zeros(len(ALL_PITCH_TYPES), dtype=np.float32)
    if pitch_type in ALL_PITCH_TYPES:
        vec[ALL_PITCH_TYPES.index(pitch_type)] = 1.0
    return vec


def build_location_features(plate_x: float, plate_z: float) -> np.ndarray:
    dist = np.sqrt(plate_x ** 2 + (plate_z - 2.5) ** 2)
    is_strike = float(abs(plate_x) <= 0.83 and 1.5 <= plate_z <= 3.5)
    return np.array([plate_x, plate_z, dist, is_strike], dtype=np.float32)


def build_context_features(df):
    stand_cols   = [c for c in df.columns if c.startswith("stand_")]
    pthrows_cols = [c for c in df.columns if c.startswith("p_throws_")]
    cols = ["balls", "strikes", "outs_when_up", "inning", "score_diff", "bat_speed_val",
            "on_1b", "on_2b", "on_3b", "has_bat_speed"] + stand_cols + pthrows_cols
    return df[cols].values.astype(float), cols


def print_bucket_distribution(labels, bucket_names, title):
    print(f"  {title}:")
    for i, name in enumerate(bucket_names):
        count = (labels == i).sum()
        print(f"    {i} {name:15s}: {count:6,}  ({100*count/len(labels):.1f}%)")


def build_dataset(df):
    df = df[df["description"].isin(BALLS_IN_PLAY)].copy()
    df = df.dropna(subset=["launch_speed", "launch_angle"])

    PT  = np.array([build_pitch_type_onehot(pt) for pt in df["pitch_type"]], dtype=np.float32)
    LOC = np.array([build_location_features(x, z) for x, z in zip(df["plate_x"], df["plate_z"])], dtype=np.float32)
    CTX, ctx_cols = build_context_features(df)
    P   = df["pitcher_id"].values.astype(np.int32)
    B   = df["batter_id"].values.astype(np.int32)
    ev_bucket = np.array([ev_to_bucket(v) for v in df["launch_speed"]], dtype=np.int32)
    

    return PT, LOC, CTX, P, B, ev_bucket, ctx_cols


if __name__ == "__main__":
    pitcher_le = pickle.load(open(SHARED_DIR + "pitcher_le.pkl", "rb"))
    batter_le  = pickle.load(open(SHARED_DIR + "batter_le.pkl", "rb"))

    print("Loading statcast data...")
    df = load_statcast()

    print("Preprocessing...")
    df = preprocess(df, pitcher_le, batter_le)

    print("Building dataset...")
    PT, LOC, CTX, P, B, ev_bucket, ctx_cols = build_dataset(df)

    print(f"\nDataset size: {len(ev_bucket):,} balls in play")
    print_bucket_distribution(ev_bucket, EV_BUCKET_NAMES, "EV bucket distribution")
    print(f"\nShapes — PT: {PT.shape}  LOC: {LOC.shape}  CTX: {CTX.shape}  ev_bucket: {ev_bucket.shape}")


    print("\nTrain/test split...")
    (PT_train,  PT_test,
    LOC_train, LOC_test,
    CTX_train, CTX_test,
    P_train,   P_test,
    B_train,   B_test,
    evb_train, evb_test,
    ) = train_test_split(
        PT, LOC, CTX, P, B, ev_bucket,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        shuffle=True
    )

    # Scale sequence and location features 
    
    # Scale location and context features
    ctx_scaler = StandardScaler()
    CTX_train_s = CTX_train.copy()
    CTX_test_s  = CTX_test.copy()
    CTX_train_s[:, :6] = ctx_scaler.fit_transform(CTX_train[:, :6])
    CTX_test_s[:,  :6] = ctx_scaler.transform(CTX_test[:, :6])

    loc_scaler = StandardScaler()
    LOC_train_s = loc_scaler.fit_transform(LOC_train)
    LOC_test_s  = loc_scaler.transform(LOC_test)

    

    print("Saving artifacts...")
    np.save(ARTIFACTS + "CTX_train.npy", CTX_train_s)
    np.save(ARTIFACTS + "CTX_test.npy",  CTX_test_s)
    np.save(ARTIFACTS + "P_train.npy",   P_train)
    np.save(ARTIFACTS + "P_test.npy",    P_test)
    np.save(ARTIFACTS + "B_train.npy",   B_train)
    np.save(ARTIFACTS + "B_test.npy",    B_test)
    np.save(ARTIFACTS + "PT_train.npy",  PT_train)
    np.save(ARTIFACTS + "PT_test.npy",   PT_test)
    np.save(ARTIFACTS + "LOC_train.npy", LOC_train_s)
    np.save(ARTIFACTS + "LOC_test.npy",  LOC_test_s)
    np.save(ARTIFACTS + "evb_train.npy", evb_train)
    np.save(ARTIFACTS + "evb_test.npy",  evb_test)


    pickle.dump(ctx_scaler,       open(ARTIFACTS + "ctx_scaler.pkl",  "wb"))
    pickle.dump(loc_scaler,       open(ARTIFACTS + "loc_scaler.pkl",  "wb"))
    pickle.dump(ctx_cols,         open(ARTIFACTS + "ctx_features.pkl","wb"))
    pickle.dump(EV_BUCKET_NAMES,  open(ARTIFACTS + "ev_buckets.pkl",  "wb"))
    pickle.dump(ALL_PITCH_TYPES,  open(ARTIFACTS + "pitch_types.pkl", "wb"))

    print("\n✓ EV/LA classification dataset built.")
    print(f"PT_train: {PT_train.shape}  CTX_train: {CTX_train_s.shape}  evb_train: {evb_train.shape}")
    print("P min/max:", P.min(), P.max())
    print("B min/max:", B.min(), B.max())