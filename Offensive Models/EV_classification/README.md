# EV Classification

Predicts the exit velocity bucket of a ball in play — Soft, Medium, Hard, or Barrel — given the pitch type, its location, the current game situation, batter bat speed, and learned profiles of both the pitcher and batter. The predicted bucket drives the rule-based classifier in `inference.py`, which combines it with a physics-based launch angle to determine the hit outcome.

---

## Files

| File | Purpose |
|---|---|
| `build_EV_dataset.py` | Loads raw Statcast CSVs, preprocesses balls in play, builds feature arrays, saves artifacts |
| `EV_architecture.py` | Keras model definition |
| `train_EV_model.py` | Trains the model, evaluates, saves `.keras` + scalers |
| `artifacts/` | All saved artifacts (model, scalers, feature lists, datasets) |

---

## Dataset Builder

**Input:** Raw Statcast CSVs (2021–2025), one per year.

**Prerequisites:** Run `build_player_vocab.py` first to generate pitcher/batter vocabulary files. Run `build_bat_speed_lookup.py` to generate `bat_speed_lookup.pkl` and `bat_speed_pop_mean.pkl` in `artifacts/shared/`.

**Scope:** Only balls in play are included (`hit_into_play`, `hit_into_play_no_out`, `hit_into_play_score`). Swings-and-misses and fouls are excluded since exit velocity is only meaningful on fair contact.

**EV bucket labels:**

| Label | Class Index | Exit Velocity Range |
|---|---|---|
| `Soft` | 0 | < 70 mph |
| `Medium` | 1 | 70–84 mph |
| `Hard` | 2 | 85–99 mph |
| `Barrel` | 3 | ≥ 100 mph |

**Preprocessing steps:**
- Drop rows missing required features
- Filter to known pitch types only
- Convert base occupancy (`on_1b`, `on_2b`, `on_3b`) to binary 0/1
- Compute `score_diff = bat_score - fld_score`
- One-hot encode batter stance and pitcher hand via `pd.get_dummies`
- Encode pitcher and batter MLBAM IDs via shared vocabulary
- Map each batter to their mean bat speed from Statcast (2024–2025). Batters not in the lookup receive the population mean (~69.57 mph) as imputation; `has_bat_speed` flag records whether the value is real or imputed

**Feature groups:**

| Group | Features | Dim |
|---|---|---|
| Pitch type | One-hot encoded pitch type | 14 |
| Location | `plate_x`, `plate_z`, `dist_to_center`, `is_strike` | 4 |
| Context | `balls`, `strikes`, `outs_when_up`, `inning`, `score_diff`, `on_1b`, `on_2b`, `on_3b`, `bat_speed_val`, `has_bat_speed`, `stand_L/R`, `p_throws_L/R` | 14 |

**Scaling:** `ctx_scaler` is a `StandardScaler` fit on only the first 6 context columns (balls, strikes, outs_when_up, inning, score_diff, on_1b). `loc_scaler` is a `StandardScaler` fit on all 4 location features. Pitch type one-hots are not scaled.

**Artifacts saved to `artifacts/`:**
- `pitch_types.pkl` — ordered list of pitch type strings
- `ev_buckets.pkl` — ordered bucket label list `["Soft", "Medium", "Hard", "Barrel"]`
- `loc_scaler.pkl` — `StandardScaler` fit on location features
- `ctx_scaler.pkl` — `StandardScaler` fit on first 6 context features
- `ctx_features.pkl` — ordered list of context feature column names
- `PT_train/test.npy`, `LOC_train/test.npy`, `CTX_train/test.npy` — feature arrays
- `P_train/test.npy`, `B_train/test.npy` — pitcher/batter ID arrays
- `evb_train/test.npy` — integer bucket labels (0–3)

---

## Model Architecture

```
Inputs:
  pitchtype_input → (batch, 14)    # one-hot pitch type
  loc_input       → (batch, 4)    # plate_x, plate_z, dist_to_center, is_strike
  context_input   → (batch, 14)   # game situation + bat speed + handedness
  pitcher_input   → (batch, 1)    # pitcher MLBAM encoded ID
  batter_input    → (batch, 1)    # batter MLBAM encoded ID

Embedding(num_pitchers, 32) → Flatten → pitcher_emb  (batch, 32)
Embedding(num_batters,  32) → Flatten → batter_emb   (batch, 32)

Concatenate([pitchtype_input, loc_input, context_input, pitcher_emb, batter_emb])
Dense(128, relu)
Dropout(0.30)
Dense(64, relu)
Dropout(0.20)
Dense(32, relu)
Dropout(0.10)
Dense(16, relu)
Dense(4, softmax)               → [P(Soft), P(Medium), P(Hard), P(Barrel)]
```

### Why these choices?

**Classification over regression:** Exit velocity as a raw number is difficult to predict accurately — regression models collapse toward the mean (~88 mph) and produce unrealistically compressed distributions. Binning into 4 outcome buckets is more tractable and is all that's needed downstream to drive the rule classifier.

**Bat speed feature:** A batter's average bat speed (from Statcast, available 2024–2025) is the closest available proxy for swing power. Batters with higher bat speeds are more likely to barrel the ball. Population mean imputation is used for batters without data, with a `has_bat_speed` flag so the model can weight real values appropriately.

**Larger embeddings (dim=32):** Pitcher and batter identity is more important for contact quality than for swing/take decisions. Pitchers with elite spin rates or unusual movement profiles consistently suppress hard contact; batters have consistent pull power tendencies. 32 dimensions gives more capacity to capture these individual profiles.

**Focal loss (γ=2.0):** The Hard class comprises ~42% of balls in play. Focal loss down-weights easy examples the model is already confident about and focuses training on the harder-to-classify Soft and Barrel ends of the distribution.

---

## Training

- **Loss:** Focal loss (γ=2.0)
- **Optimizer:** Adam (lr=0.0001, clipnorm=1.0)
- **Epochs:** Up to 50, with `EarlyStopping(patience=6)` on `val_loss`
- **LR schedule:** `ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6)`
- **Best val accuracy:** 43.3% (epoch 2), restored via `restore_best_weights=True`
- **Split:** 85/15 train/test, with 20% of train used for validation
- **Batch size:** 256

---

## Performance

| Metric | Value |
|---|---|
| Test Accuracy | 43.0% |
| Precision (Soft) | 0.51 |
| Recall (Soft) | 0.02 |
| Precision (Medium) | 0.35 |
| Recall (Medium) | 0.15 |
| Precision (Hard) | 0.44 |
| Recall (Hard) | 0.85 |
| Precision (Barrel) | 0.47 |
| Recall (Barrel) | 0.17 |
| Macro F1 | 0.27 |
| Weighted F1 | 0.35 |

```
              precision    recall  f1-score   support

        Soft       0.51      0.02      0.04     11107
      Medium       0.35      0.15      0.21     22310
        Hard       0.44      0.85      0.58     40669
      Barrel       0.47      0.17      0.24     23028

    accuracy                           0.43     97114
   macro avg       0.44      0.30      0.27     97114
weighted avg       0.43      0.43      0.35     97114

Confusion matrix:
[[  246  2561  8047   253]
 [  124  3372 17879   935]
 [   98  3059 34384  3128]
 [   14   709 18501  3804]]
```

**Context:** A random baseline would achieve 25% accuracy on this 4-class problem; the model reaches 43%. The dominant limitation is Hard-class bias — Hard contact comprises ~42% of training samples and achieves 85% recall, while Soft (2% recall) and Barrel (17% recall) are rarely predicted. This reflects the fundamental ceiling imposed by the available features: pitch type, location, count, and bat speed do not strongly predict contact quality in isolation. The information that would matter most — swing path, contact point, and attack angle — is not available in public Statcast data. For the simulator, this means most balls in play will be classified as Hard, which is partially offset by the physics-based launch angle sampling producing varied trajectories for the rule classifier.

---

## Artifacts Consumed Downstream

The predicted EV bucket (`"Soft"`, `"Medium"`, `"Hard"`, or `"Barrel"`) is combined with a physics-sampled launch angle in `inference.py`'s rule-based classifier to determine the hit outcome (single, double, home run, flyout, groundout, lineout, or popup) on each fair ball.
