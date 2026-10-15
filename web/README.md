# Microgrid / RL — interactive dashboard

A single-page dashboard over the simulation results in this repo: an hour-by-hour
episode replay, head-to-head agent comparison, and an interactive forecast-error
sweep.

Every number on the page is read from `public/data/simulation.json`, which is
generated from the actual trained models. Nothing is hardcoded.

## Running it

```bash
# 1. train the agents (from the repo root) — several hours on CPU/GPU
python main.py

# 2. export results to JSON
python export_web_data.py

# 3. serve the dashboard
cd web
npm install
npm run dev
```

Then open http://localhost:5173.

The dashboard works without step 1: `export_web_data.py` falls back to the LP
benchmark and the two baseline agents, and the robustness section shows a
placeholder until trained models exist.

## Build

```bash
npm run build     # -> dist/
npm run preview
```

`dist/` is a static bundle — deploy it to GitHub Pages, Netlify or Vercel as-is.
For a project page served from a subpath, set `base` in `vite.config.ts`.

## Regenerating data

`export_web_data.py` (repo root) writes `web/public/data/simulation.json`. Re-run
it after retraining. It picks up whichever of `sac_model.zip`, `rsac_model.zip`
and `ppo_model.zip` exist in `results/`, so partial runs still produce a valid
page.

Tunables at the top of that script:

| Constant | Meaning |
|---|---|
| `N_EVAL_EPISODES` | episodes in the evaluation sweep (initial SOC 0.2 → 0.8) |
| `TRAJECTORY_INITIAL_SOC` | starting charge for the replayed episode |

## Layout

```
src/
├── data/          types, fetch hook, agent colours and TOU bands
├── components/    nav, shared chart primitives (tooltip, legend, TOU shading)
└── sections/      Hero · Replay · Comparison · Robustness · System
```

Charts are Recharts. Colour, spacing and type live as CSS custom properties in
`src/styles/tokens.css` — change the palette there and it propagates to both the
charts and the UI.
