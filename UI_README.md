# Organization Threat Surface Simulator - Web UI

A web-based interface for exploring organizational misalignment and shadow structure formation using Streamlit.

## Features

### 🚀 Quick Demo Mode
- Run simulations on synthetic organizations with adjustable parameters
- Visualize reward dynamics over time
- View organization structure with reported/edict/shadow link differentiation
- Browse action logs with filtering
- Inspect individual agent behaviors

### 🎛️ Custom Simulation Mode
- Use different data sources: Synthetic, Snapshot, or Master SP500 TMT
- Fine-tune cost parameters, friction, and agent weights
- Run single organizations with full analysis

### 📊 Batch Analysis Mode
- Run multiple simulations across many organizations
- Analyze aggregate dynamics and shadow link formation
- Compute confidence intervals for results

### 🔍 Data Explorer Mode
- Browse SP500 Top Management Teams data
- Explore snapshot organization data
- View and analyze simulation results

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements_ui.txt
   ```

2. **Verify data files exist:**
   ```bash
   ls -la master_SP500_TMT.csv snapshot.csv
   ```

## Running the App

### Start the web server:
```bash
streamlit run Simulations/app.py
```

The app will open in your default browser at `http://localhost:8501`

### Alternative (with custom config):
```bash
streamlit run Simulations/app.py --logger.level=info
```

## Usage Guide

### Quick Demo Walkthrough

1. **Select "Quick Demo" from the sidebar**
2. **Adjust parameters:**
   - Number of Agents: 10-100
   - Hierarchy Depth: 1-5 levels
   - Simulation Steps: 10-200
   - Mean Personal Weight: 0-1 (higher = more misalignment)
   - Advanced options for fine-tuning costs

3. **Click "🚀 Run Simulation"**

4. **Explore Results:**
   - **📊 Dynamics:** Time series of organizational vs personal rewards
   - **🕸️ Graph:** Visual network showing organization structure
     - Gray edges: Normal reporting relationships
     - Red edges: Edicts (direct orders)
     - Red dotted edges: Shadow links (suspicious, long-distance shortcuts)
   - **📋 Action Log:** Complete record of all agent actions with filtering
     - Filter by action type (email/edict)
     - Filter by agent type (normal/misaligned)
     - Filter by shadow link status
   - **👥 Agent Details:** Individual agent inspection
     - View gain weights and behavioral profile
     - See all actions taken by the agent
     - Inspect score components for each action

### Custom Simulation

1. Select data source: Synthetic, Snapshot, or Master (SP500)
2. Configure:
   - Steps per simulation
   - Maximum nodes per organization
   - Agent parameters (personal weight distribution)
   - Cost parameters (friction, edict cost, hop cost)
3. Run and view results

### Data Explorer

- **Master SP500 TMT:** Browse top management team data with company/year filtering
- **Snapshot:** View organization snapshots
- **Simulation Results:** Access previously saved results

## Understanding the Visualizations

### Reward Dynamics Chart
- **Blue line:** Organizational score (alignment with mission)
- **Orange line:** Personal score (misaligned agents pursuing personal targets)
- Rising personal score with stable organizational score = hidden misalignment
- Divergence indicates organizational dysfunction

### Organization Graph
- **Node size:** Relative influence (betweenness centrality)
- **Gray arrows:** Normal hierarchical communication
- **Red arrows:** Edict edges (direct orders)
- **Red dotted arrows:** Shadow links (suspicious bypasses > 3 levels)

### Action Log Metrics
- **Cost:** Token cost of the action
- **Utility:** Perceived benefit to the agent
- **Shadow Link:** Whether this creates a suspicious structural shortcut

## Key Parameters Explained

| Parameter | Meaning | Impact |
|-----------|---------|--------|
| Mean Personal Weight | Average personal vs org motivation | Higher = more misalignment |
| Friction | Cost multiplier for upward communication | Higher = more edicts |
| Base Edict Cost | Fixed cost of creating direct shortcuts | Higher = fewer edicts |
| Utility Threshold | Minimum benefit to justify edict | Higher = fewer edicts |

## Interpreting Results

### Red Flags (Shadow Structure Formation)
- ✅ Presence of edict edges (red arrows)
- ✅ Long-distance shadow links (red dotted arrows)
- ✅ Rising personal score while org score stagnates
- ✅ Misaligned agents (M_ prefix) issuing multiple edicts

### Healthy Organization
- ✅ Mostly gray edges (normal hierarchy)
- ✅ Few edicts
- ✅ Organizational and personal scores track together
- ✅ Normal agents making rational decisions

## Technical Notes

- Simulations are CPU-bound; large orgs (>100 nodes) may take time
- Results are stored in session state; refresh page to reset
- Batch mode uses multiprocessing; scales with available cores
- Graph layouts use spring layout with edge weights for positioning

## Troubleshooting

### App won't start
```bash
# Check Streamlit installation
pip list | grep streamlit

# Reinstall if needed
pip install --upgrade streamlit
```

### Data files not found
```bash
# Ensure you're in the Simulations directory
cd /Users/mp/org-threat-surface/Simulations

# Check file existence
ls -la *.csv
```

### Slow simulations
- Reduce number of agents/steps
- Use Quick Demo for testing
- Batch mode uses multiprocessing; check CPU availability

### Memory issues
- Reduce max_nodes
- Reduce repeats in batch mode
- Clear browser cache and restart

## Extending the UI

### Adding custom modes:
Edit `app.py` and add new `elif mode == "My Mode":` blocks

### Adding new visualizations:
Use Plotly for interactive charts or Matplotlib for static ones

### Saving results:
```python
# In app.py, add export functionality
df_logs.to_csv("my_simulation_results.csv", index=False)
```

## References

- Streamlit docs: https://docs.streamlit.io
- Plotly docs: https://plotly.com/python/
- NetworkX docs: https://networkx.org/
- Simulation model: `main.py` in this directory
