#!/usr/bin/env python3
"""
Flask server to serve organization graph data for the interactive dashboard.
"""
import os
import json
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import networkx as nx
import numpy as np
import random

# Import from main.py
import sys
sys.path.insert(0, os.path.dirname(__file__))
from main import build_org_from_master, SimulationConfig, EdgeType

app = Flask(__name__)
CORS(app)

# Configuration
MASTER_PATH = os.path.join(os.path.dirname(__file__), "master_SP500_TMT.csv")
RNG = np.random.default_rng(42)

def _infer_level_from_snapshot_row(row: pd.Series) -> int:
    """Infer hierarchy level from a data row."""
    role = str(row.get("role", "")).lower()
    title = str(row.get("title", "")).lower()
    
    # CEO level
    if "ceo" in role or "chief executive" in title:
        return 0
    
    # C-suite level
    if any(x in role for x in ["cfo", "coo", "cto", "cmo", "chief"]):
        return 1
    
    # VP level
    if "vp" in role or "vice president" in title:
        return 2
    
    # Director/Manager level
    if any(x in role for x in ["director", "manager", "head"]):
        return 3
    
    # Fallback
    return 3

@app.route('/')
def index():
    """Serve the dashboard HTML."""
    return send_from_directory(os.path.dirname(__file__), 'dashboard.html')

@app.route('/results/<path:filename>')
def serve_results(filename):
    """Serve CSV files from results directory."""
    return send_from_directory(os.path.join(os.path.dirname(__file__), 'results'), filename)

@app.route('/api/graph/<int:gv_key>/<int:year>')
def get_graph(gv_key: int, year: int):
    """Build and return graph structure for a given GV_KEY and Year."""
    try:
        # Create a minimal config
        config = SimulationConfig(
            num_agents=50,
            max_depth=4,
        )
        
        # Build the organization
        org = build_org_from_master(
            config=config,
            master_path=MASTER_PATH,
            gv_key=gv_key,
            year=year,
            max_nodes=50,  # Limit for visualization
        )
        
        # Convert to JSON-serializable format
        nodes = []
        for node_id, attrs in org.graph.nodes(data=True):
            nodes.append({
                "id": node_id,
                "role": attrs.get("role", "Unknown"),
                "level": attrs.get("level", 0),
                "centrality": float(attrs.get("centrality", 0.0)),
            })
        
        edges = []
        for source, target, attrs in org.graph.edges(data=True):
            edges.append({
                "source": source,
                "target": target,
                "weight": float(attrs.get("weight", 1.0)),
                "type": attrs.get("type", "reporting"),
            })
        
        return jsonify({
            "gv_key": gv_key,
            "year": year,
            "nodes": nodes,
            "edges": edges,
            "num_nodes": len(nodes),
            "num_edges": len(edges),
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/companies')
def get_companies():
    """Get list of unique companies from misalignment data."""
    try:
        misalignment_path = os.path.join(os.path.dirname(__file__), "results", "misalignment_potential.csv")
        if not os.path.exists(misalignment_path):
            return jsonify({"companies": [], "years": []})
        
        df = pd.read_csv(misalignment_path)
        companies = sorted(df["GV_KEY"].unique().tolist())
        years = sorted(df["Year"].unique().tolist())
        
        return jsonify({
            "companies": companies,
            "years": years,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting dashboard server...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)

