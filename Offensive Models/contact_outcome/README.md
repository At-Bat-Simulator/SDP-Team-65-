# Contact Outcome Prediction

Predicts the outcome of a swing — miss, foul, or fair ball — given the pitch type, its location, the current game situation, and learned profiles of both the pitcher and batter. Outputs a probability distribution over three classes used by the simulator to stochastically determine what happens when a batter swings.

---

## Files

| File | Purpose |
|---|---|
| `build_contact_dataset.py` | Loads raw Statcast CSVs, preprocesses swing pitches, builds feature arrays, saves artifacts |
| `contact_architecture.py` | Keras model definition |
| `train_contact_model.py` | Trains the model, evaluates, saves `.keras` + scalers |
| `artifacts/` | All saved artifacts (model, scalers, feature lists, datasets) |

---

## Dataset Builder

**Input:** Raw Statcast CSVs (2021–2025), one per year.

**Prerequisites:** Run `build_player_vocab.py` first to generate the pitcher/batter vocabulary files used during ID encoding.

**Scope:** Only swing pitches are included. Takes (called strikes, balls) are filtered out before training since the contact outcome question is only meaningful given that the batter has already swung.

**Contact outcome labels:**

| Label | Class Index | Statcast Descriptions |
|---|---|---|
| `miss` | 0 | `swinging_strike`, `swinging_strike_blocked` |
| `foul` | 1 | `foul`, `foul_tip` |
| `fair` | 2 | `hit_into_play`, `hit_into_play_no_out`, `hit_into_play_score` |

**Preprocessing steps:**
- Drop rows with missing required features
- Filter to swing descriptions only
- Convert base occupancy (`on_1b`, `on_2b`, `on_3b`) to binary 0/1
- Compute `score_diff = bat_score - fld_score`
- One-hot encode batter stance (`stand_L`, `stand_R`) and pitcher hand (`p_throws_L`, `p_throws_R`)
- Filter to known pitch types only
- Encode pitcher and batter MLBAM IDs as integer category codes via shared vocabulary

**Feature groups:**

| Group | Features | Dim |
|---|---|---|
| Pitch type | One-hot encoded pitch type | 14 |
| Location | `plate_x`, `plate_z`, `dist_to_center`, `is_strike` | 4 |
| Context | `balls`, `strikes`, `outs_when_up`, `inning`, `score_diff`, `on_1b`, `on_2b`, `on_3b`, `stand_L/R`, `p_throws_L/R` | 12 |

**Artifacts saved to `artifacts/`:**
- `pitch_types.pkl` — ordered list of pitch type strings used for one-hot encoding
- `classes.pkl` — ordered class label list `["miss", "foul", "fair"]`
- `loc_scaler.pkl` — `StandardScaler` fit on location features
- `ctx_scaler.pkl` — `StandardScaler` fit on the 5 continuous context features
- `ctx_features.pkl` — ordered list of context feature column names
- `PT_train/test.npy`, `LOC_train/test.npy`, `CTX_train/test.npy` — feature arrays
- `P_train/test.npy`, `B_train/test.npy` — pitcher/batter ID arrays
- `y_train/test.npy` — integer class labels (0=miss, 1=foul, 2=fair)

---

## Model Architecture

```
Inputs:
  pitchtype_onehot → (batch, 14)    # one-hot pitch type
  location_features → (batch, 4)   # plate_x, plate_z, dist_to_center, is_strike
  context_features  → (batch, 12)  # game situation + handedness
  pitcher_id        → (batch, 1)   # pitcher MLBAM encoded ID
  batter_id         → (batch, 1)   # batter MLBAM encoded ID

Embedding(num_pitchers, 8) → Flatten → pitcher_emb  (batch, 8)
Embedding(num_batters,  8) → Flatten → batter_emb   (batch, 8)

Concatenate([pitchtype_onehot, location_features, context_features, pitcher_emb, batter_emb])
Dense(128, relu)
Dropout(0.30)
Dense(64, relu)
Dropout(0.20)
Dense(32, relu)
Dropout(0.10)
Dense(3, softmax)                   → [P(miss), P(foul), P(fair)]
```

### Why these choices?

**Same feedforward structure as swing/take:** Contact outcome shares the same input signal as the swing/take decision — pitch type, location, count, and player identity. A feedforward network over the current pitch's features is appropriate here; no sequential context is needed to determine what happens on a given swing.

**Three separate feature groups:** Pitch type, location, and game context are scaled independently (or left as-is for binary features). This prevents high-variance continuous features from dominating.

**Embeddings (dim=8):** Pitcher and batter embeddings capture individual tendencies — some batters make more consistent contact, some pitchers generate more swings-and-misses on specific pitch types. 8 dimensions is sufficient for this 3-class task.

**Tapered dropout (0.30 → 0.20 → 0.10):** Aggressive regularization early in the network that relaxes as layers narrow, preventing overfitting on the larger early layers while allowing fine-grained use of the learned representation.

**Softmax output:** Outputs a probability distribution over [miss, foul, fair]. The simulator samples from this distribution to stochastically determine what happens when the batter swings, producing realistic variation rather than always predicting the most likely class.

---

## Training

- **Loss:** Sparse categorical cross-entropy
- **Optimizer:** Adam (lr=0.0001, clipnorm=1.0)
- **Epochs:** Up to 20, with `EarlyStopping(patience=3)` on `val_loss`
- **Best weights restored from:** Epoch 5 (val_loss 0.9863), stopped at epoch 8
- **Split:** 85/15 train/test, with 20% of train used for validation
- **Batch size:** 256

---

## Performance

| Metric | Value |
|---|---|
| Test Accuracy | 49.0% |
| Precision (miss) | 0.57 |
| Recall (miss) | 0.43 |
| Precision (foul) | 0.46 |
| Recall (foul) | 0.59 |
| Precision (fair) | 0.50 |
| Recall (fair) | 0.42 |
| Macro F1 | 0.49 |
| Weighted F1 | 0.49 |

```
              precision    recall  f1-score   support

        miss       0.57      0.43      0.49     63183
        foul       0.46      0.59      0.51    105508
        fair       0.50      0.42      0.46     97658

    accuracy                           0.49    266349
   macro avg       0.51      0.48      0.49    266349
weighted avg       0.50      0.49      0.49    266349

Confusion matrix:
 [[27158 26118  9907]
  [12303 62211 30994]
  [ 8582 48172 40904]]
```

**Context:** This is a 3-class problem, so a random baseline would achieve ~33% accuracy. The model reaches 49%, a meaningful improvement. Foul/fair confusion is the dominant error source and is expected — 48,172 fair balls are predicted as foul, and 30,994 fouls are predicted as fair. This is structurally expected: foul balls and fair balls are physically similar (same contact zone, similar pitch types), differing only in spray direction, which is not observable in the pre-contact features available to the model. Miss precision (0.57) is the strongest of the three classes because swings-and-misses are more tightly tied to pitch type and location — off-speed out of the zone, elevated fastball. The model is not biased toward any single class.

---

## Artifacts Consumed Downstream

The contact outcome probability vector `[P(miss), P(foul), P(fair)]` is used by the simulator at inference time — `inference.py` calls this model on each swing pitch and samples from the distribution to determine whether the batter misses, fouls off, or puts the ball in play. This drives realistic count progression and fair-ball rate throughout the at-bat.
