import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from EVLA_architecture import build_ev_model

ART_PATH   = "artifacts/"
SHARED_DIR = "../../artifacts/shared/"


def evaluate(evb_true, ev_pred_probs, lab_true, la_pred_probs, ev_bucket_names, la_bucket_names):
    ev_pred = np.argmax(ev_pred_probs, axis=1)
    la_pred = np.argmax(la_pred_probs, axis=1)

    ev_acc = np.mean(ev_pred == evb_true)
    la_acc = np.mean(la_pred == lab_true)

    print(f"\n  Exit Velocity Bucket Accuracy : {ev_acc*100:.1f}%")
    print("\n  EV Classification Report:")
    print(classification_report(evb_true, ev_pred, target_names=ev_bucket_names, zero_division=0))
    print("  EV Confusion Matrix:")
    print(confusion_matrix(evb_true, ev_pred))

    print(f"\n  Launch Angle Bucket Accuracy  : {la_acc*100:.1f}%")
    print("\n  LA Classification Report:")
    print(classification_report(lab_true, la_pred, target_names=la_bucket_names, zero_division=0))
    print("  LA Confusion Matrix:")
    print(confusion_matrix(lab_true, la_pred))


if __name__ == "__main__":
    print("Loading EV/LA dataset artifacts...")
    CTX_train = np.load(ART_PATH + "CTX_train.npy")
    CTX_test  = np.load(ART_PATH + "CTX_test.npy")
    P_train   = np.load(ART_PATH + "P_train.npy")
    P_test    = np.load(ART_PATH + "P_test.npy")
    B_train   = np.load(ART_PATH + "B_train.npy")
    B_test    = np.load(ART_PATH + "B_test.npy")
    PT_train  = np.load(ART_PATH + "PT_train.npy")
    PT_test   = np.load(ART_PATH + "PT_test.npy")
    LOC_train = np.load(ART_PATH + "LOC_train.npy")
    LOC_test  = np.load(ART_PATH + "LOC_test.npy")
    evb_train = np.load(ART_PATH + "evb_train.npy")
    evb_test  = np.load(ART_PATH + "evb_test.npy")
    lab_train = np.load(ART_PATH + "lab_train.npy")
    lab_test  = np.load(ART_PATH + "lab_test.npy")

    ev_bucket_names = pickle.load(open(ART_PATH + "ev_buckets.pkl", "rb"))
    la_bucket_names = pickle.load(open(ART_PATH + "la_buckets.pkl", "rb"))
    pitcher_le      = pickle.load(open(SHARED_DIR + "pitcher_le.pkl", "rb"))
    batter_le       = pickle.load(open(SHARED_DIR + "batter_le.pkl",  "rb"))

    num_features   = CTX_train.shape[1]
    pitch_type_dim = PT_train.shape[1]
    loc_dim        = LOC_train.shape[1]
    num_pitchers   = len(pitcher_le.classes_)
    num_batters    = len(batter_le.classes_)
    num_ev_buckets = len(ev_bucket_names)
    num_la_buckets = len(la_bucket_names)

    
    print(f"loc_dim={loc_dim}  num_pitchers={num_pitchers}  num_batters={num_batters}")
    print(f"num_ev_buckets={num_ev_buckets}  num_la_buckets={num_la_buckets}")

    # Class weights — penalize mistakes on rarer buckets more heavily
    ev_weights = compute_class_weight("balanced", classes=np.unique(evb_train), y=evb_train)
    la_weights = compute_class_weight("balanced", classes=np.unique(lab_train), y=lab_train)
    ev_class_weight = dict(enumerate(ev_weights))
    la_class_weight = dict(enumerate(la_weights))
    print("\nEV class weights:", {ev_bucket_names[k]: f"{v:.2f}" for k, v in ev_class_weight.items()})
    print("LA class weights:", {la_bucket_names[k]: f"{v:.2f}" for k, v in la_class_weight.items()})

    model = build_ev_model(
        num_features=num_features,
        num_pitchers=num_pitchers,
        num_batters=num_batters,
        pitch_type_dim=pitch_type_dim,
        loc_dim=loc_dim,
        num_ev_buckets=num_ev_buckets,
        num_la_buckets=num_la_buckets,
    )

    model.compile(
        optimizer="adam",
        loss={
            "ev_out": "sparse_categorical_crossentropy",
            "la_out": "sparse_categorical_crossentropy",
        },
        loss_weights={"ev_out": 1.0, "la_out": 1.0},
        metrics={"ev_out": "accuracy", "la_out": "accuracy"},
    )
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1),
    ]

    # Compute per-sample weights by combining EV and LA class weights
    ev_sample_weights = np.array([ev_weights[c] for c in evb_train])
    la_sample_weights = np.array([la_weights[c] for c in lab_train])

    # Average the two so each sample has one combined weight
    combined_weights = (ev_sample_weights + la_sample_weights) / 2.0

    history = model.fit(
    [PT_train, LOC_train, CTX_train, P_train, B_train],
    [evb_train, lab_train],  # list instead of dict
    sample_weight=combined_weights,
    validation_split=0.2,
    epochs=50,
    batch_size=256,
    callbacks=callbacks,
    verbose=1
    )

    print("\nEvaluating on test set...")
    ev_pred_probs, la_pred_probs = model.predict(
    [PT_test, LOC_test, CTX_test, P_test, B_test],
    batch_size=256, verbose=0
    )
    evaluate(evb_test, ev_pred_probs, lab_test, la_pred_probs, ev_bucket_names, la_bucket_names)

    model.save(ART_PATH + "ev_model.keras")
    print("\nSaved model:", ART_PATH + "ev_model.keras")