# Frontend

React + Vite single-page application that serves as the user interface for the At-Bat Simulator. Provides a home page for selecting pitcher/batter matchups and a simulator page that drives the pitch-by-pitch at-bat experience.

---

## Pages

| Page | Route | Description |
|---|---|---|
| `HomePage.jsx` | `/` | Pitcher and batter selection, matchup entry point |
| `SimulatorPage.jsx` | `/simulator` | Pitch-by-pitch at-bat simulator with live zone visualization |
| `ModelDocs.jsx` | `/docs` | Model documentation viewer |

### HomePage
Allows the user to select a pitcher and batter from dropdown menus populated by the `/api/players` endpoint. Passes the selected matchup to the simulator via React Router state.

### SimulatorPage
The core simulator interface. On each pitch:
- Calls `POST /api/predict` with the current pitcher, batter, and count state
- Animates the ball traveling from the pitcher to the strike zone
- Displays pitch type, velocity, location, and result
- Maintains a running at-bat history with color-coded outcomes
- Shows real historical matchup data (last real at-bat) in the right panel
- Displays exit velocity, launch angle, and hit type on fair ball contact

---

## Running Locally

```bash
cd frontend
npm install
npm run dev
# Runs at http://localhost:5173
```

The frontend proxies API calls to `http://localhost:5000`. The Flask API must be running before using the simulator — see `api/README.md` for setup instructions.

---

## Building for Production

```bash
npm run build
# Output is written to dist/
```

To deploy to the VM after building:

```bash
# Run from project root on your local machine
scp frontend/dist/* car21031@atbatsimulator.engr.uconn.edu:/var/www/atbatsimulator/
```

Then SSH into the VM and fix permissions:

```bash
sudo chmod -R 755 /var/www/atbatsimulator/
```

---

## Project Structure

```
frontend/
├── src/
│   ├── pages/
│   │   ├── HomePage.jsx        # Matchup selection
│   │   ├── SimulatorPage.jsx   # At-bat simulator
│   │   └── ModelDocs.jsx       # Documentation viewer
│   ├── HomePage.css
│   ├── SimulatorPage.css
│   └── main.jsx                # React entry point, router setup
├── index.html
├── vite.config.js
└── package.json
```

---

## API Dependency

The frontend expects the Flask API to be running at `http://localhost:5000` in development and proxied via nginx at the same domain in production. All API calls use relative paths (`/api/*`) which nginx forwards to `http://127.0.0.1:5000`.

See `api/README.md` for full endpoint documentation.
