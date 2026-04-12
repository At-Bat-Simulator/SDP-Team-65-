from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dense, Dropout,
    Concatenate, RepeatVector, BatchNormalization
)
from tensorflow.keras.models import Model


def build_ev_model(
    seq_len: int,
    num_features: int,
    num_pitchers: int,
    num_batters: int,
    pitch_type_dim: int,
    loc_dim: int,
    num_ev_buckets: int = 4,
    num_la_buckets: int = 5,
):
    """
    Predicts exit velocity bucket (4 classes) and launch angle bucket (4 classes).
    Both heads share the same LSTM 

    Inputs:
      - seq_input:       (seq_len, num_features)
      - pitcher_input:   ()
      - batter_input:    ()
      - pitchtype_input: (pitch_type_dim,)
      - loc_input:       (loc_dim,)

    Outputs:
      - ev_out: (num_ev_buckets,)  softmax — Soft / Medium / Hard / Barrel
      - la_out: (num_la_buckets,)  softmax — Groundball / LD / Fly Ball / Popup
    """
    # Sequence branch 
    seq_input     = Input(shape=(seq_len, num_features), name="seq_input")
    pitcher_input = Input(shape=(),                      name="pitcher_input", dtype="int32")
    batter_input  = Input(shape=(),                      name="batter_input",  dtype="int32")

    pitcher_emb = Embedding(num_pitchers, 32, name="pitcher_emb")(pitcher_input)
    batter_emb  = Embedding(num_batters,  32, name="batter_emb")(batter_input)

    p_rep = RepeatVector(seq_len)(pitcher_emb)
    b_rep = RepeatVector(seq_len)(batter_emb)

    x = Concatenate(axis=-1)([seq_input, p_rep, b_rep])
    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.20)(x)
    x = LSTM(64)(x)
    x = Dropout(0.20)(x)
    x = Dense(32, activation="relu")(x)

    # Target pitch branch 
    pitchtype_input = Input(shape=(pitch_type_dim,), name="pitchtype_input")
    loc_input       = Input(shape=(loc_dim,),        name="loc_input")

    shared = Concatenate()([x, pitchtype_input, loc_input])
    shared = Dense(64, activation="relu")(shared)
    shared = BatchNormalization()(shared)
    shared = Dropout(0.20)(shared)
    shared = Dense(32, activation="relu")(shared)
    shared = Dropout(0.10)(shared)

    # EV head: 4-class classification 
    ev = Dense(16, activation="relu")(shared)
    ev_out = Dense(num_ev_buckets, activation="softmax", name="ev_out")(ev)

    # LA head: 4-class classification
    la = Dense(16, activation="relu")(shared)
    la_out = Dense(num_la_buckets, activation="softmax", name="la_out")(la)

    return Model(
        inputs=[seq_input, pitcher_input, batter_input, pitchtype_input, loc_input],
        outputs=[ev_out, la_out]
    )