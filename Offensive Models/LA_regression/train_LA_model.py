import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from LA_architecture import build_la_model

ART_PATH   = "artifacts/"
SHARED_DIR = "../../artifacts/shared/"


def evaluate(y_true_s, y_pred_s, target_scaler):
    y_true = target_scaler.inverse_transform(y_true_s)
    y_pred = target_scaler.inverse_transform(y_pred_s)
    mae    = np.mean(np.abs(y_true - y_pred))
    print(f"\n  Launch Angle MAE: {mae:.2f} degrees")
    print(f"  Predicted — mean: {y_pred.mean():.1f}  std: {y_pred.std():.1f}  "
          f"min: {y_pred.min():.1f}  max: {y_pred.max():.1f}")
    print(f"  Actual    — mean: {y_true.mean():.1f}  std: {y_true.std():.1f}  "
          f"min: {y_true.min():.1f}  max: {y_true.max():.1f}")
    return mae


if __name__ == "__main__":
    print("Loading LA regression dataset artifacts...")
    PT_train  = np.load(ART_PATH + "PT_train.npy")
    PT_test   = np.load(ART_PATH + "PT_test.npy")
    LOC_train = np.load(ART_PATH + "LOC_train.npy")
    LOC_test  = np.load(ART_PATH + "LOC_test.npy")
    CTX_train = np.load(ART_PATH + "CTX_train.npy")
    CTX_test  = np.load(ART_PATH + "CTX_test.npy")
    P_train   = np.load(ART_PATH + "P_train.npy")
    P_test    = np.load(ART_PATH + "P_test.npy")
    B_train   = np.load(ART_PATH + "B_train.npy")
    B_test    = np.load(ART_PATH + "B_test.npy")
    y_train   = np.load(ART_PATH + "y_train.npy")
    y_test    = np.load(ART_PATH + "y_test.npy")

    target_scaler = pickle.load(open(ART_PATH + "target_scaler.pkl", "rb"))
    pitcher_le    = pickle.load(open(SHARED_DIR + "pitcher_le.pkl",  "rb"))
    batter_le     = pickle.load(open(SHARED_DIR + "batter_le.pkl",   "rb"))

    num_features   = CTX_train.shape[1]
    pitch_type_dim = PT_train.shape[1]
    loc_dim        = LOC_train.shape[1]
    num_pitchers   = len(pitcher_le.classes_)
    num_batters    = len(batter_le.classes_)

    print(f"num_features={num_features}  pitch_type_dim={pitch_type_dim}  loc_dim={loc_dim}")
    print(f"num_pitchers={num_pitchers}  num_batters={num_batters}")

    model = build_la_model(
        num_features=num_features,
        num_pitchers=num_pitchers,
        num_batters=num_batters,
        pitch_type_dim=pitch_type_dim,
        loc_dim=loc_dim,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0),
        loss="mse",
        metrics=["mae"],
    )
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        [PT_train, LOC_train, CTX_train, P_train, B_train],
        y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=256,
        callbacks=callbacks,
        verbose=1,
    )

    print("\nEvaluating on test set...")
    y_pred = model.predict(
        [PT_test, LOC_test, CTX_test, P_test, B_test],
        batch_size=256, verbose=0,
    )
    evaluate(y_test, y_pred, target_scaler)

    model.save(ART_PATH + "la_model.keras")
    print("\nSaved model:", ART_PATH + "la_model.keras")
