# Quick Start Guide - 2 Minutes to Dashboard

## 🚀 Get It Running (Fastest Route)

```bash
# Navigate to project
cd /Users/mp/org-threat-surface

# Run the quick-start script
./start_ui.sh
```

**That's it!** Browser opens to `http://localhost:8501`

## 🎮 Your First Simulation (30 seconds)

1. **Click "Run Simulation"** button (left side shows default parameters)
2. **Wait** for results (usually 5-30 seconds)
3. **Explore tabs:**
   - 📊 **Dynamics** - See how scores change over time
   - 🕸️ **Graph** - Visualize the organization structure
   - 📋 **Log** - Browse all agent actions
   - 👥 **Details** - Inspect individual agents

## 🎯 What You're Looking For

### ✅ Signs of Organizational Health
- Mostly gray edges (normal chain of command)
- Few red edges (minimal shadow commands)
- Organizational and personal scores moving together

### ⚠️ Red Flags (Problem Signals)
- Many red edges (excessive edicts)
- Red dotted lines (suspicious long-distance shortcuts)
- Personal score rising while org score stagnates
- Misaligned agents (names starting with `M_`) taking many actions

## 🎛️ Experiment: Try These Settings

### Test 1: High Misalignment (Chaos)
1. **Mean Personal Weight**: Drag to 0.8
2. Run Simulation
3. Compare: Notice way more edicts and shadow links

### Test 2: Low Misalignment (Order)
1. **Mean Personal Weight**: Drag to 0.1
2. Run Simulation
3. Compare: Much fewer edicts, stable scores

### Test 3: Large Organization (Complexity)
1. **Number of Agents**: Set to 75
2. **Hierarchy Depth**: Set to 4
3. Run Simulation
4. View Graph tab to see structure complexity

## 📊 Understanding the Charts

### Reward Dynamics (Dynamics Tab)
- **Blue line** = Organizational alignment (good)
- **Orange line** = Personal interests (selfish)
- If they **diverge** = organizational dysfunction
- If they **track together** = healthy alignment

### Organization Graph (Graph Tab)
Colors & styles:
- 🔘 **Nodes** = People
- **Gray arrows** → Normal reporting
- **Red arrows** → Direct orders (edicts)
- **Red dotted** → Shadow links (suspicious)

## 📥 What Each Column Means (Log Tab)

| Column | Meaning |
|--------|---------|
| **Step** | When in simulation |
| **Agent** | Who took action (M_ = misaligned) |
| **Action** | email or edict |
| **Cost** | Token cost to agent |
| **Utility** | Benefit agent gained |
| **Shadow Link** | ✓ = suspicious bypass |

## ⚡ Power Tips

- **Fast Testing**: Reduce "Simulation Steps" to 20
- **See Edicts**: Increase "Base Edict Cost" slider
- **More Misalignment**: Drag "Mean Personal Weight" right
- **Bigger Orgs**: Increase agents and depth

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Port already in use" | Kill existing: `lsof -ti:8501 \| xargs kill` |
| "Module not found" | Run: `pip install -r Simulations/requirements_ui.txt` |
| "Slow simulations" | Reduce agents and steps |
| "Can't see data" | Check: `ls Simulations/*.csv` |

## 📚 Learn More

| Want To... | See... |
|-----------|--------|
| Understand full features | `UI_README.md` |
| Deploy to cloud | `DEPLOYMENT.md` |
| See technical details | `UI_SUMMARY.md` |
| Understand the model | `main.py` in Simulations/ |

## 🔄 Workflow Examples

### Example 1: Detect When Shadow Structures Form
1. Start with **Personal Weight = 0.1**
2. Gradually increase in steps of 0.1
3. At each level, run simulation and count red dotted edges
4. **Finding**: Shadow links appear around 0.5-0.7

### Example 2: Analyze Different Org Structures
1. Use "Custom Simulation" mode
2. Select **"Snapshot"** or **"Master (SP500)"** data source
3. Run simulation on real org data
4. Compare dynamics vs synthetic orgs

### Example 3: Deep Dive on Single Agent
1. Run Quick Demo
2. Go to "Agent Details" tab
3. Select a misaligned agent (M_xxxx)
4. Examine their decisions and score components

## 💾 Save Your Results

After running simulation:
- **Copy data**: Right-click table → Copy
- **Save image**: Right-click graph → Save image
- **Export CSV**: Tables can be copied to Excel

## 🎓 Understanding the Simulation

**Simple Version:**
- Agents make decisions to achieve personal goals vs org goals
- They can send messages (emails) through hierarchy OR issue orders (edicts) that bypass it
- When misaligned agents create too many edicts, "shadow structures" form
- This makes the org harder to manage and less aligned

**What's Being Measured:**
- How much do agents care about org mission vs personal gain? (weights)
- How many shortcuts do they create? (edicts)
- Which shortcuts are suspicious? (shadow links > 3 levels)
- Does organizational performance degrade? (diverging scores)

## 🎓 Key Concepts

| Term | Means |
|------|-------|
| **Edict** | Direct order that bypasses normal hierarchy |
| **Shadow Link** | Suspicious edict (jumps over 3+ hierarchy levels) |
| **Misaligned** | Agent prioritizes personal goals (marked M_) |
| **Normal** | Agent aligned with org goals (marked A_) |
| **Tokens** | "Budget" agents spend on communications |
| **Utility** | Benefit agent perceived from their action |

## ✨ Next Level: Modify Code

Want to customize? Edit `Simulations/app.py`:

```python
# Change default values (line ~60)
num_agents = st.slider("Number of Agents", 10, 100, 50)  # Default now 50

# Change visualization colors (search "color=")
nx.draw_networkx_edges(..., edge_color='green', ...)

# Add new mode (search "elif mode ==")
elif mode == "My Mode":
    st.header("Custom Mode")
    # Add your code here
```

Then refresh browser - changes take effect immediately!

---

**Ready?** Run `./start_ui.sh` and start exploring! 🚀
