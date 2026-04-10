# Swing/Take Prediction

Predicts whether a batter will swing at a given pitch given the pitch type, its location, the current game situation, and learned profiles of both the pitcher and batter. Outputs a continuous swing probability used by the simulator to stochastically decide swing vs. take on each pitch.

---

## Files

| File | Purpose |
|---|---|
| `build_swingtake_dataset.py` | Loads raw Statcast CSVs, preprocesses, builds feature arrays, saves artifacts |
| `swingtake_architecture.py` | Keras model definition |
| `train_swingtake_model.py` | Trains the model, evaluates, saves `.keras` + scalers |
| `artifacts/` | All saved artifacts (model, scalers, feature lists, datasets) |

---

## Dataset Builder

**Input:** Raw Statcast CSVs (2021–2025), one per year.

**Prerequisites:** Run `build_player_vocab.py` first to generate the pitcher/batter vocabulary files used during ID encoding.

**Swing label:** A pitch is labeled as a swing (`1`) if its Statcast description is one of:
```
swinging_strike, swinging_strike_blocked, foul, foul_tip,
hit_into_play, hit_into_play_no_out, hit_into_play_score
```
All other descriptions (called strike, ball, etc.) are labeled as takes (`0`).

**Preprocessing steps:**
- Drop rows with missing required features
- Convert base occupancy (`on_1b`, `on_2b`, `on_3b`) to binary 0/1
- Compute `score_diff = bat_score - fld_score`
- One-hot encode batter stance (`stand_L`, `stand_R`) and pitcher hand (`p_throws_L`, `p_throws_R`)
- Filter to known pitch types only
- Encode pitcher and batter MLBAM IDs as integer category codes via shared vocabulary

**Feature groups:**

Unlike the pitch type and location models, the swing/take model does not use a sequence window. Each pitch is treated as an independent sample with three feature groups:

| Group | Features | Dim |
|---|---|---|
| Pitch type | One-hot encoded pitch type | 14 |
| Location | `plate_x`, `plate_z`, `dist_to_center`, `is_strike` | 4 |
| Context | `balls`, `strikes`, `outs_when_up`, `inning`, `score_diff`, `on_1b`, `on_2b`, `on_3b`, `stand_L/R`, `p_throws_L/R` | 12 |

**Artifacts saved to `artifacts/`:**
- `pitch_types.pkl` — ordered list of pitch type strings used for one-hot encoding
- `loc_scaler.pkl` — `StandardScaler` fit on location features
- `ctx_scaler.pkl` — `StandardScaler` fit on continuous context features only (binary features are not scaled)
- `ctx_features.pkl` — ordered list of context feature column names
- `ctx_n_continuous.pkl` — integer count of continuous columns (used to apply scaler only to the continuous slice)
- `PT_train/test.npy`, `LOC_train/test.npy`, `CTX_train/test.npy` — feature arrays
- `PIT_train/test.npy`, `BAT_train/test.npy` — pitcher/batter ID arrays
- `y_train/test.npy` — binary swing labels

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
Dense(1, sigmoid)                   → swing probability ∈ [0, 1]
```

### Why these choices?

**No sequence input:** Swing/take decisions are primarily driven by the current pitch — its type, location, and the count — not by the sequence of prior pitches. A feedforward network over the current pitch's features is more appropriate here than an LSTM.

**Three separate feature groups:** Pitch type, location, and game context are each scaled independently (or left as-is for binary features). This prevents high-variance continuous features from dominating.

**Embeddings (dim=8):** Pitcher and batter embeddings capture individual tendencies — some batters chase breaking balls out of the zone, some pitchers induce more takes on borderline pitches. 8 dimensions is sufficient for this binary task.

**Tapered dropout (0.30 → 0.20 → 0.10):** Aggressive regularization early in the network that relaxes as layers narrow. This prevents the larger early layers from overfitting while letting the final layers make fine-grained use of the learned representation.

**Sigmoid output:** Outputs a probability rather than a hard 0/1 prediction. The simulator uses this probability directly to make a stochastic swing/take decision on each pitch, producing realistic variation rather than always swinging or always taking.

---

## Training

- **Loss:** Binary cross-entropy
- **Optimizer:** Adam (lr=0.0001, clipnorm=1.0)
- **Epochs:** Up to 20, with `EarlyStopping(patience=3)` on `val_loss`
- **Split:** 85/15 train/test, with 20% of train used for validation
- **Batch size:** 256

---

## Performance

| Metric | Value |
|---|---|
| Test Accuracy | 73.2% |
| Precision (take) | 0.75 |
| Recall (take) | 0.74 |
| Precision (swing) | 0.72 |
| Recall (swing) | 0.73 |
| Macro F1 | 0.73 |

```
              precision    recall  f1-score   support

           0       0.75      0.74      0.74    294124
           1       0.72      0.73      0.72    266496

    accuracy                           0.73    560620
   macro avg       0.73      0.73      0.73    560620
weighted avg       0.73      0.73      0.73    560620

Confusion matrix:
 [[217733  76391]
 [ 72899 193597]]
```

**Context:** Swing/take decisions are influenced by factors the model cannot observe — pitch movement, spin rate, the batter's in-game approach, and pitcher tendencies on a given day. 73% accuracy reflects a realistic ceiling for this task given the available features. Performance is balanced across both classes, indicating the model is not biased toward always predicting swing or always predicting take.

---

## Artifacts Consumed Downstream

The swing probability output is used directly by the simulator at inference time — `inference.py` calls the swing/take model on each predicted pitch and uses the returned probability to stochastically decide whether the batter swings. This drives realistic count progression throughout the at-bat.
