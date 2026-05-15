const BASE = import.meta.env.VITE_API_URL ?? "";

export async function getPlayers() {
  const res = await fetch(`${BASE}/api/players`);
  if (!res.ok) throw new Error(`players failed: ${res.status}`);
  return res.json();
}

export async function predict(pitcher_mlbam, batter_mlbam) {
  const res = await fetch(`${BASE}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pitcher_mlbam, batter_mlbam }),
  });
  if (!res.ok) throw new Error(`predict failed: ${res.status}`);
  return res.json();
}
