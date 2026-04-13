import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

from EVLA_architecture import build_ev_model

ART_PATH   = "artifacts/"
SHARED_DIR = "../../artifacts/shared/"


def evaluate(evb_true, ev_pred_probs, ev_bucket_names):
    ev_pred = np.argmax(ev_pred_probs, axis=1)
    ev_acc = np.mean(ev_pred == evb_true)
    
    print(f"\n  Exit Velocity Bucket Accuracy : {ev_acc*100:.1f}%")
    print("\n  EV Classification Report:")
    print(classification_report(evb_true, ev_pred, target_names=ev_bucket_names, zero_division=0))
    print("  EV Confusion Matrix:")
    print(confusion_matrix(evb_true, ev_pred))


def focal_loss(gamma=2.0, alpha=None):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_onehot = tf.one_hot(y_true, depth=y_pred.shape[-1])
        
        # Clip for numerical stability
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        
        # Cross entropy
        ce = -tf.reduce_sum(y_true_onehot * tf.math.log(y_pred), axis=-1)
        
        # Probability of true class
        p_t = tf.reduce_sum(y_true_onehot * y_pred, axis=-1)
        
        # Focal weight — down-weights easy examples
        focal_weight = tf.pow(1.0 - p_t, gamma)
        
        return tf.reduce_mean(focal_weight * ce)
    return loss_fn


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

    ev_bucket_names = pickle.load(open(ART_PATH + "ev_buckets.pkl", "rb"))
    pitcher_le      = pickle.load(open(SHARED_DIR + "pitcher_le.pkl", "rb"))
    batter_le       = pickle.load(open(SHARED_DIR + "batter_le.pkl",  "rb"))

    num_features   = CTX_train.shape[1]
    pitch_type_dim = PT_train.shape[1]
    loc_dim        = LOC_train.shape[1]
    num_pitchers   = len(pitcher_le.classes_)
    num_batters    = len(batter_le.classes_)
    num_ev_buckets = len(ev_bucket_names)


    
    print(f"loc_dim={loc_dim}  num_pitchers={num_pitchers}  num_batters={num_batters}")


    

    model = build_ev_model(
        num_features=num_features,
        num_pitchers=num_pitchers,
        num_batters=num_batters,
        pitch_type_dim=pitch_type_dim,
        loc_dim=loc_dim,
        num_ev_buckets=num_ev_buckets,
        
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0),
        loss=focal_loss(gamma=2.0),
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
    [PT_train, LOC_train, CTX_train, P_train, B_train],
    evb_train,  # list instead of dict
    validation_split=0.2,
    epochs=50,
    batch_size=256,
    callbacks=callbacks,
    verbose=1
    )

    print("\nEvaluating on test set...")
    ev_pred_probs = model.predict(
    [PT_test, LOC_test, CTX_test, P_test, B_test],
    batch_size=256, verbose=0
    )
    evaluate(evb_test, ev_pred_probs, ev_bucket_names)

    model.save(ART_PATH + "ev_model.keras")
    print("\nSaved model:", ART_PATH + "ev_model.keras")