# Org Threat Surface Toy Model

Toy simulation of an organizational social graph to explore misaligned AI behavior via token-vs-topology trade-offs.

## Installing

In order of basedness:

### Use direnv

Install direnv and nix.

Run `direnv allow`.

Use [nix-direnv](https://github.com/nix-community/nix-direnv) for faster shells.

### Use nix develop

Install nix.

Run `nix develop`.

### Use UV directly

Install UV.

Run `uv sync`.

## Running

Prefix commands with `uv run`, like:

```
uv run main.py --mode sweep --n-orgs 100 --repeats 10 --steps 100
```

## Interactive Dashboard

To launch the interactive dashboard:

1. Make sure you have run simulations and generated results in `results/`:
   - `results/misalignment_potential.csv`
   - `results/sweep_summary.csv`

2. Start the dashboard server:
   ```bash
   uv run dashboard_server.py
   ```

3. Open your browser to `http://localhost:5000`

The dashboard provides:
- **Sweep Results Visualization**: Interactive charts showing reward dynamics vs. personal weight with confidence intervals
- **Organization Filtering**: Filter and sort organizations by year, misalignment metrics, etc.
- **Network Visualization**: Click any organization to see its network structure with a physics-based force-directed layout
- **Demo-Ready Aesthetics**: Modern, polished UI perfect for presentations
