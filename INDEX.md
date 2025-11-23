# Documentation Index

## 🚀 Getting Started (Start Here!)

### **[QUICK_START.md](QUICK_START.md)** ⭐ START HERE
- **2-minute setup guide**
- First simulation walkthrough
- What to look for (red flags vs health)
- Simple troubleshooting
- **Best for**: First-time users wanting fast results

---

## 📚 Main Documentation

### **[UI_README.md](UI_README.md)**
- Complete feature overview
- Installation & setup
- Detailed usage guide
- All modes explained
- Parameter reference table
- Visualization explanations
- Troubleshooting
- **Best for**: Learning all features, understanding what's possible

### **[UI_SUMMARY.md](UI_SUMMARY.md)**
- Architecture overview
- Complete file listing
- Feature breakdown
- Deployment options table
- Extension points
- Known limitations
- **Best for**: Understanding what was built and how

### **[DEPLOYMENT.md](DEPLOYMENT.md)**
- Local development
- Docker deployment
- Cloud options (Streamlit Cloud, Heroku, AWS, Google Cloud, Kubernetes)
- Production setup (SSL/TLS, auth, rate limiting)
- Performance tuning
- Monitoring & health checks
- **Best for**: Deploying to production or cloud

---

## 🛠️ Technical Files

### **Simulations/app.py**
- Main Streamlit application (450 lines)
- 4 modes: Quick Demo, Custom, Batch, Data Explorer
- All UI components and logic
- **Best for**: Customizing interface, adding features

### **Simulations/main.py**
- Simulation engine (1500+ lines)
- OrganizationModel, Agent, decision logic
- Cost calculations, edict effects
- Data generation from real/synthetic orgs
- **Best for**: Understanding the simulation math

### **Simulations/requirements_ui.txt**
- Python dependencies for web UI
- **Best for**: Installing dependencies locally

### **start_ui.sh**
- One-command startup script
- Checks prerequisites, installs deps, launches server
- **Best for**: Quick launch on macOS/Linux

### **Dockerfile**
- Container configuration
- Pre-installed dependencies
- Optimized for cloud deployment
- **Best for**: Docker/containerized deployments

### **docker-compose.yml**
- Multi-container orchestration
- Volume mounts, port config
- Auto-restart policy
- **Best for**: One-command Docker deployment

---

## 📊 Data Files (in Simulations/)

### **master_SP500_TMT.csv** (50MB)
- SP500 Top Management Teams data
- 10+ years of executive records
- Real organizational hierarchies
- Used by Master data source

### **snapshot.csv**
- Organization snapshot data
- Single-point-in-time org structures
- Used by Snapshot data source

### **results/** (directory)
- Output directory for simulation results
- CSV exports, analysis files
- Persists between runs

---

## 🎯 Navigation by Use Case

### "I want to use it RIGHT NOW"
1. [QUICK_START.md](QUICK_START.md) (5 min read)
2. Run `./start_ui.sh`
3. Click "Run Simulation"
4. Explore tabs

### "I want to understand all features"
1. [UI_README.md](UI_README.md) (15 min read)
2. Try each mode
3. Read parameter explanations
4. Experiment with settings

### "I want to deploy this"
1. [DEPLOYMENT.md](DEPLOYMENT.md) - choose your platform
2. Follow platform-specific instructions
3. Test locally first
4. Deploy to production

### "I want to customize/extend"
1. [UI_SUMMARY.md](UI_SUMMARY.md) - understand architecture
2. Edit `Simulations/app.py`
3. Add features (new modes, visualizations, export)
4. Refresh browser to see changes

### "I want to understand the science"
1. Read comments in `Simulations/main.py`
2. Understand agent decision logic (lines 467-619)
3. Study cost calculations (lines 158-213)
4. Review edict mechanics (lines 215-228)

---

## 🔍 Quick Reference

### File Organization
```
org-threat-surface/
├── README.md                 ← Original project README
├── QUICK_START.md           ← YOU ARE HERE - 2 min start
├── UI_README.md             ← Complete feature guide
├── UI_SUMMARY.md            ← Architecture & overview
├── DEPLOYMENT.md            ← Deployment to cloud
├── INDEX.md                 ← This file
├── Dockerfile               ← Container config
├── docker-compose.yml       ← Docker orchestration
├── start_ui.sh              ← Quick launch script
│
└── Simulations/
    ├── main.py              ← Simulation engine
    ├── app.py               ← Streamlit web UI
    ├── requirements_ui.txt   ← Python deps
    ├── master_SP500_TMT.csv ← Real org data
    ├── snapshot.csv         ← Org snapshots
    └── results/             ← Output directory
```

### Command Quick Guide
```bash
# Start UI (fastest way)
./start_ui.sh

# Manual start
pip install -r Simulations/requirements_ui.txt
cd Simulations && streamlit run app.py

# Docker start
docker-compose up

# Run CLI simulations (no UI)
cd Simulations && python main.py --mode single --steps 100

# Kill if port is stuck
lsof -ti:8501 | xargs kill
```

---

## 🎓 Learning Paths

### Beginner (30 min)
1. Read: [QUICK_START.md](QUICK_START.md)
2. Do: Run first simulation
3. Explore: All four tabs
4. Result: Understand basics

### Intermediate (2-3 hours)
1. Read: [UI_README.md](UI_README.md)
2. Try: Custom Simulation mode
3. Experiment: Different parameters
4. Analyze: Real org data vs synthetic
5. Result: Know all features

### Advanced (1 day+)
1. Read: [DEPLOYMENT.md](DEPLOYMENT.md)
2. Study: `main.py` code
3. Deploy: Docker or cloud
4. Modify: `app.py` to add features
5. Extend: Custom modes or visualizations
6. Result: Production-ready system

---

## ❓ FAQ

**Q: Where do I start?**
A: [QUICK_START.md](QUICK_START.md) then run `./start_ui.sh`

**Q: How do I customize it?**
A: Edit `Simulations/app.py`, changes appear on browser refresh

**Q: Can I deploy to cloud?**
A: Yes! See [DEPLOYMENT.md](DEPLOYMENT.md) for 6 options

**Q: What's the performance impact of large orgs?**
A: >100 nodes may be slow. See DEPLOYMENT.md > Performance Tuning

**Q: Can multiple users access it simultaneously?**
A: Yes with Streamlit Cloud or multi-container deployment

**Q: How do I save results?**
A: Right-click tables/graphs to copy/export; results auto-saved to CSV

**Q: What's the difference between modes?**
A: See [UI_README.md](UI_README.md) under "Features" section

---

## 📞 Support Resources

| Need | Resource |
|------|----------|
| Quick help | This INDEX.md |
| Getting started | QUICK_START.md |
| Feature guide | UI_README.md |
| Technical details | UI_SUMMARY.md |
| Deployment | DEPLOYMENT.md |
| Simulation math | main.py comments |
| UI code | app.py |

---

## 📦 What's Included

✅ **Web Application**: Interactive Streamlit UI  
✅ **Simulation Engine**: Agent-based org model  
✅ **Real Data**: SP500 TMT + snapshots  
✅ **Visualizations**: Network graphs + time series  
✅ **Documentation**: 5 guides + this index  
✅ **Deployment**: Docker + cloud configs  
✅ **Quick Start**: 1-command launch  

---

## 🎯 Next Steps

1. **Immediate**: Run `./start_ui.sh`
2. **Today**: Explore all modes, try different parameters
3. **This week**: Read deployment docs if interested
4. **This month**: Customize for your use case
5. **Ongoing**: Share feedback and improvements

---

**Questions?** Check the relevant doc above. **Ready?** Start with [QUICK_START.md](QUICK_START.md)! 🚀
