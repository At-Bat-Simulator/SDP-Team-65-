# scripts / 

Utility scripts for data preparation, model support, and simulation validation. 
These scripts are run from the **project root** unless otherwise noted

---

## Files


### `build_player_vocab.py`
**Run before training any model**

Reads all Statcast CSVs (2021-2025) and fits `LabelEncoder` objects that map each pitcher and batter MLBAM ID to an integer embedding index.
All five models depend on these shared encoders to keep player embedding indices consistent

**Outputs** -> `artifacts/shared/pitcher_le.pkl`, `artifacts/shared/batter_le.pkl`

```bash
python scripts/build_player_vocab.py
```



### `build_player_names.py`

Looks up human-readable player metadata for every pitcher and batter in the dataset using pybaseball.playerid_reverse_lookup. Stores name,
handedness (bats/throws), pitch count, plate appearance count, and strike zone dimensions (sz_top, sz_bot) per batter

**Outputs** -> `artifacts/shared/player_names.json`

```bash
python scripts/build_player_names.py
```




### `build_serving_table.py`

Builds the inference-time serving table from raw Statcast CSVs. Performs all feature engineering needed at prediction time:
    * Encodes base occupancy, handedness, previous pitch type as one-hot columns
    * Computes per-player average exit velocity and launch angle
    * Computes previous pitch location within each at-bat
    * Computes per-batter spray direction tendencies (pull% / center%/ oppo%)
    * Assigns pitcher and batter embedding indices from shared encoders
    * Sorts chronologically for sequence windowing

**Outputs** -> `artifacts/serving/serving_table.parquet`
    Note: This file is not committed to git (>100MB). It must be rebuilt locally and on the VM whenever this script changes

```bash
python scripts/build_serving_table.py
```



### `batch_sim.py`

Simulates 500 at-bats by calling the live Flask API at http://localhost:5000, implements a full at-bat state machine (swing/take decisions,
count tracking, terminal conditions) and prints an MLB-comparable statistical report including:
    * Strikeout / walk / ball-in-play rates
    * Average pitches per plate appearance
    * First pitch strike rate
    * Pitch type distribution
    * Swing rate, contact rate, whiff rate
    * Count distribution
Used for validating simulation realism. Flask must be running before executing

```bash
python scripts/batch_sim.py
```






