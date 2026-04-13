from tensorflow.keras.layers import (
    Input, Embedding, Dense, Dropout,
    Concatenate, Flatten
)
from tensorflow.keras.models import Model


def build_ev_model(
    num_features: int,
    num_pitchers: int,
    num_batters: int,
    pitch_type_dim: int,
    loc_dim: int,
    num_ev_buckets: int = 4,
):
    pitcher_input   = Input(shape=(1,), name="pitcher_input", dtype="int32")
    batter_input    = Input(shape=(1,), name="batter_input",  dtype="int32")
    pitchtype_input = Input(shape=(pitch_type_dim,), name="pitchtype_input")
    loc_input       = Input(shape=(loc_dim,),        name="loc_input")
    context_input   = Input(shape=(num_features,),   name="context_input")

    p_emb = Flatten()(Embedding(num_pitchers, 32, name="pitcher_emb")(pitcher_input))
    b_emb = Flatten()(Embedding(num_batters,  32, name="batter_emb")(batter_input))

    x = Concatenate()([pitchtype_input, loc_input, context_input, p_emb, b_emb])
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.30)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.20)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.10)(x)

    ev = Dense(16, activation="relu")(x)
    ev_out = Dense(num_ev_buckets, activation="softmax", name="ev_out")(ev)

    return Model(
        inputs=[pitchtype_input, loc_input, context_input, pitcher_input, batter_input],
        outputs=[ev_out]
    )
