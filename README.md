# SDP Team 65 — Baseball At-Bat Simulator

A pitch-by-pitch at-bat simulator built on real MLB Statcast data (2021–2025). Given a pitcher and batter, a pipeline of five machine learning models predicts pitch type, location, swing decision, contact outcome, and exit velocity/launch angle — simulating a realistic at-bat through an interactive web interface.


---

## Project Structure

```
SDP-Team-65-/
├── Pitch Type Prediction/       # LSTM model: predicts next pitch type
├── Pitch Location Prediction/   # LSTM + GMM: predicts plate_x / plate_z
├── Offensive Models/   
    |- swingtake/                # Dense network: predicts P(swing)
    |- contact_outcome/          # Dense network: predicts foul/miss/fair
    |- EV_and_LA/                # LSTM: predicts exit velocity + launch angle
├── api/                         # Flask backend serving all model predictions
|-- frontend/                    # React/Vite frontend
├── csv data/                    # Raw Statcast CSVs (not committed to repo)
├── artifacts/shared/            # Shared player encoders (pitcher_le, batter_le)
|-- artifacts/serving/           # Serving table (not committed, rebuild on server)
└── scripts/                     # Data prep, serving table, batch simulation
```

---

## Setup

### Requirements
- Python 3.10+
- TensorFlow 2.x
- scikit-learn, pandas, numpy
- pybaseball (for downloading Statcast data)
- Flask, flask-cors

```bash
pip install tensorflow scikit-learn pandas numpy pybaseball flask flask-cors
```

### Downloading Data
Raw Statcast CSVs live in `csv data/`. To download 2021–2024:
```bash
python "csv data/download_full_statcast.py"
```

---

## Training Pipeline (one-time)

Run these in order. Each step produces artifacts consumed by the next.

```bash

#0. Build shared player vocab first (required by all models)
python scripts/build_player_vocab.py
python scripts/build_player_names.py

# 1. Pitch type model
cd "Pitch Type Prediction"
python build_pitchtype_dataset.py
python train_pitchtype_model.py      # saves pitchtype_model.keras + pitch_type_probs.npy

# 2. Pitch location model
cd "../Pitch Location Prediction"
python build_pitchlocation_dataset.py
python train_pitchlocation_model.py  # saves pitch_location_model.keras
python fit_location_gmm.py           # saves location_gmm.pkl (must run after training)

# 3. Swing/take model
cd "../Offensive Models"
python build_swingtake_dataset.py
python train_swingtake_model.py

# 4. Contact outcome model
cd "../contact_outcome"
python build_contact_dataset.py
python train_contact_model.py

# 5. EV/LA model
cd "../EV_and_LA"
python build_EVLA_dataset.py
python3 train_EVLA_model.py

# 6. Rebuild serving table (also required on VM after changes)
cd ../..
python scripts/build_serving_table.py



## Running the App

### Start the API
```bash
cd api
python app.py
# Runs at http://localhost:5000
```

### Start the Frontend
```bash
# From the frontend directory
npm install
npm run dev
# Runs at http://localhost:5173
```

---

## Models Overview

| Model | Type | Output |
|---|---|---|
| Pitch Type | LSTM + Player Embeddings | Softmax over pitch types (FF, SL, CU, ...) |
| Pitch Location | LSTM + Gaussian NLL + GMM | plate_x, plate_z (feet from center of plate) |
| Swing/Take    | Dense + Player Embeddings   |  P(swing) sigmoid | 
| Contact Outcome | Dense + Player Embeddings | Softmax: foul / miss / fair |
| Exit Velocity & Launch Angle | LSTM + Player Embeddings | EV (mph), LA (degrees) | 
---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/players` | GET | Returns list of valid pitchers and batters |
| `/api/predict` | POST | Given pitcher + batter + game context, returns next pitch prediction |
| `/api/matchup-history` | GET | Returns most recent real at-bat between pitcher and batter from statcast 

See `api/README.md` for full request/response format.

---


## Inference Pipeline

Each call to `/api/predict` runs the following sequence:

1. **Serving table lookup** - fetch historical pitch sequence for the matchup (falls back to pitcher-only -> global if sparse)
2. **Pitch Type** -- LSTM predicts pitch type, masked to pitcher's real repertoire
3. **Location** -- LSTM predicts mean location, adjusted by count-based targeting bias, sampled from real Statcast GMM
4. **Swing/Take** -- Dense model predicts swing probability given pitch type + location + game context
5. **Contact Outcome** -- Dense model predicts foul / miss / fair if batter swings
6. **EV/LA** -- If fair contact, LSTM predicts exit velocity and launch angle
7. **Hit Type**-- Rule-based classifier uses EV + LA + batter spray tendency -> single / double / HR / groundout / flyout / popup

> **Note:** `artifacts/serving/serving_table.parquet` is not committed to git (>100MB)
> It must be rebuilt locally and on the VM by running `python scripts/build_serving_table.py`
> before starting the API

## Team
SDP Team 65 — Senior Design Project
