import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import os
import sys
from typing import Optional, Tuple, List
import networkx as nx
from io import StringIO

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from main import (
    OrganizationModel,
    SimulationConfig,
    Agent,
    ActionLog,
    run_simulation,
    compute_step_reward_series,
    build_org_from_master,
    build_org_from_snapshot,
    ActionType,
    PsychProfile,
)

st.set_page_config(
    page_title="Org Threat Surface Simulator",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏢 Organization Threat Surface Simulator")
st.markdown("""
Explore how misaligned agents create shadow structures within organizational hierarchies.
""")

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================
st.sidebar.header("⚙️ Configuration")

mode = st.sidebar.radio(
    "Simulation Mode",
    ["Quick Demo", "Custom Simulation", "Batch Analysis", "Data Explorer", "CSV Visualizer"]
)

# ============================================================================
# MODE: QUICK DEMO
# ============================================================================
if mode == "Quick Demo":
    st.header("Quick Demo: Synthetic Organization")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Parameters")
        num_agents = st.slider("Number of Agents", 10, 100, 30)
        max_depth = st.slider("Hierarchy Depth", 1, 5, 3)
        steps = st.slider("Simulation Steps", 10, 200, 50)
        mean_personal_weight = st.slider("Mean Personal Weight", 0.0, 1.0, 0.3, 0.1)
    
    with col2:
        st.subheader("Advanced Options")
        friction = st.slider("Communication Friction", 0.0, 2.0, 0.5, 0.1)
        utility_threshold = st.slider("Edict Utility Threshold", 0.0, 0.5, 0.1, 0.05)
        base_edict_cost = st.slider("Base Edict Cost", 5, 50, 20)
    
    if st.button("🚀 Run Simulation", key="demo_run"):
        with st.spinner("Running simulation..."):
            config = SimulationConfig(
                num_agents=num_agents,
                max_depth=max_depth,
                friction=friction,
                base_edict_cost=base_edict_cost,
                utility_threshold=utility_threshold,
                mean_personal_weight=mean_personal_weight,
            )
            
            org, agents, logs = run_simulation(
                steps=steps,
                data_source="synthetic",
                mean_personal_weight=mean_personal_weight,
            )
            
            # Store in session state
            st.session_state.org = org
            st.session_state.agents = agents
            st.session_state.logs = logs
            st.session_state.steps = steps
            st.success("Simulation complete!")
    
    # Display results if available
    if "org" in st.session_state:
        org = st.session_state.org
        logs = st.session_state.logs
        steps = st.session_state.steps
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        total_actions = len([l for l in logs if l.action_type != ActionType.NONE])
        emails = len([l for l in logs if l.action_type == ActionType.EMAIL])
        edicts = len([l for l in logs if l.action_type == ActionType.EDICT])
        suspicious_edicts = len(org.detect_suspicious_edicts())
        
        with col1:
            st.metric("Total Actions", total_actions)
        with col2:
            st.metric("Emails", emails)
        with col3:
            st.metric("Edicts", edicts)
        with col4:
            st.metric("Suspicious Edicts", suspicious_edicts)
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Dynamics",
            "🕸️ Graph",
            "📋 Action Log",
            "👥 Agent Details"
        ])
        
        with tab1:
            st.subheader("Reward Dynamics Over Time")
            org_series, pers_series = compute_step_reward_series(logs, steps)
            
            fig = go.Figure()
            steps_x = np.arange(len(org_series))
            
            fig.add_trace(go.Scatter(
                x=steps_x, y=org_series, mode='lines+markers',
                name='Organizational Score', line=dict(color='blue', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=steps_x, y=pers_series, mode='lines+markers',
                name='Personal Score', line=dict(color='orange', width=2)
            ))
            
            fig.update_layout(
                title="Average Rewards Per Step",
                xaxis_title="Step",
                yaxis_title="Score",
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Organization Structure")
            
            # Create networkx visualization
            fig, ax = plt.subplots(figsize=(12, 8))
            
            pos = nx.spring_layout(org.graph, seed=42, weight="weight", k=2)
            
            # Separate edge types
            reporting_edges = [
                (u, v) for u, v, d in org.graph.edges(data=True)
                if d.get("type") == "REPORTING"
            ]
            edict_edges = [
                (u, v) for u, v, d in org.graph.edges(data=True)
                if d.get("type") == "EDICT"
            ]
            
            suspicious_set = set(org.detect_suspicious_edicts())
            shadow_edges = [e for e in edict_edges if e in suspicious_set]
            normal_edicts = [e for e in edict_edges if e not in suspicious_set]
            
            # Draw nodes
            nx.draw_networkx_nodes(
                org.graph, pos, node_color='lightblue',
                node_size=300, ax=ax
            )
            
            # Draw edges
            nx.draw_networkx_edges(
                org.graph, pos, edgelist=reporting_edges,
                edge_color='gray', alpha=0.5, arrows=True, ax=ax
            )
            
            if normal_edicts:
                nx.draw_networkx_edges(
                    org.graph, pos, edgelist=normal_edicts,
                    edge_color='red', width=2, arrows=True, ax=ax
                )
            
            if shadow_edges:
                nx.draw_networkx_edges(
                    org.graph, pos, edgelist=shadow_edges,
                    edge_color='red', style='dotted', width=2.5, arrows=True, ax=ax
                )
            
            nx.draw_networkx_labels(org.graph, pos, font_size=8, ax=ax)
            ax.set_title("Organization Structure\n(Gray: Reporting | Red: Edicts | Red Dotted: Shadow Links)")
            ax.axis('off')
            st.pyplot(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Action Log")
            
            # Convert logs to dataframe
            log_data = []
            for log in logs:
                if log.action_type != ActionType.NONE:
                    log_data.append({
                        "Step": log.step,
                        "Agent": log.agent_id,
                        "Action": log.action_type.value,
                        "Source": log.source,
                        "Target": log.target,
                        "Cost": f"{log.cost:.2f}",
                        "Utility": f"{log.utility:.2f}",
                        "Shadow Link": "✓" if log.shadow_link else "✗",
                    })
            
            df_logs = pd.DataFrame(log_data)
            
            # Filters
            col1, col2, col3 = st.columns(3)
            with col1:
                action_filter = st.multiselect(
                    "Filter by Action",
                    [ActionType.EMAIL.value, ActionType.EDICT.value],
                    default=[ActionType.EMAIL.value, ActionType.EDICT.value]
                )
            with col2:
                agent_filter = st.multiselect(
                    "Filter by Agent Type",
                    ["All", "Normal", "Misaligned"],
                    default="All"
                )
            with col3:
                shadow_filter = st.multiselect(
                    "Filter by Shadow Link",
                    ["All", "Shadow Links Only", "Non-Shadow Only"],
                    default="All"
                )
            
            # Apply filters
            df_filtered = df_logs[df_logs["Action"].isin(action_filter)]
            if agent_filter != "All":
                if agent_filter == "Normal":
                    df_filtered = df_filtered[~df_filtered["Agent"].str.startswith("M_")]
                else:
                    df_filtered = df_filtered[df_filtered["Agent"].str.startswith("M_")]
            
            if shadow_filter == "Shadow Links Only":
                df_filtered = df_filtered[df_filtered["Shadow Link"] == "✓"]
            elif shadow_filter == "Non-Shadow Only":
                df_filtered = df_filtered[df_filtered["Shadow Link"] == "✗"]
            
            st.dataframe(df_filtered, use_container_width=True, height=400)
        
        with tab4:
            st.subheader("Agent Details")
            
            agent_names = [a.id for a in st.session_state.agents]
            selected_agent = st.selectbox("Select Agent", agent_names)
            
            agent = next((a for a in st.session_state.agents if a.id == selected_agent), None)
            if agent:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Role", agent.role)
                    st.metric("Node ID", agent.node_id)
                with col2:
                    st.metric("Profile", agent.psych_profile.value)
                    st.metric("Remaining Tokens", agent.tokens)
                with col3:
                    st.metric("Org Weight", f"{agent.gain_weights['org']:.3f}")
                    st.metric("Personal Weight", f"{agent.gain_weights['personal']:.3f}")
                
                st.write("Global Weight:", f"{agent.gain_weights['global']:.3f}")
                if agent.secret_target:
                    st.write("Secret Target:", agent.secret_target)
                
                # Agent's actions
                agent_logs = [l for l in logs if l.agent_id == selected_agent and l.action_type != ActionType.NONE]
                if agent_logs:
                    st.subheader("Agent Actions")
                    for log in agent_logs[:10]:  # Show last 10
                        with st.expander(f"Step {log.step}: {log.action_type.value}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Target:** {log.target}")
                                st.write(f"**Cost:** {log.cost:.2f}")
                                st.write(f"**Utility:** {log.utility:.2f}")
                            with col2:
                                st.write(f"**Org Score:** {log.s_org:.3f}")
                                st.write(f"**Personal Score:** {log.s_personal:.3f}")
                                st.write(f"**Global Score:** {log.s_global:.3f}")

# ============================================================================
# MODE: CUSTOM SIMULATION
# ============================================================================
elif mode == "Custom Simulation":
    st.header("🎛️ Custom Simulation")
    
    data_source = st.radio(
        "Data Source",
        ["Synthetic", "Snapshot", "Master (SP500)"],
        horizontal=True
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Basic Parameters")
        steps = st.number_input("Simulation Steps", 10, 500, 100)
        max_nodes = st.number_input("Max Nodes", 5, 200, 30)
    
    with col2:
        st.subheader("Agent Parameters")
        mean_personal_weight = st.slider("Mean Personal Weight", 0.0, 1.0, 0.3)
        token_replenish = st.number_input("Token Replenish", 1, 20, 5)
        max_tokens = st.number_input("Max Tokens", 50, 500, 100)
    
    with col3:
        st.subheader("Cost Parameters")
        friction = st.slider("Friction", 0.0, 2.0, 0.5)
        base_edict_cost = st.number_input("Base Edict Cost", 5, 100, 20)
        hop_cost = st.slider("Hop Cost", 0.1, 5.0, 1.0)
    
    if st.button("🚀 Run Custom Simulation", key="custom_run"):
        try:
            with st.spinner("Running simulation..."):
                org, agents, logs = run_simulation(
                    steps=steps,
                    data_source=data_source.lower().split()[0],
                    max_nodes=max_nodes,
                    mean_personal_weight=mean_personal_weight,
                )
                
                st.session_state.org = org
                st.session_state.agents = agents
                st.session_state.logs = logs
                st.session_state.steps = steps
                st.success("✅ Simulation complete!")
        except Exception as e:
            st.error(f"❌ Simulation failed: {str(e)}")

# ============================================================================
# MODE: BATCH ANALYSIS
# ============================================================================
elif mode == "Batch Analysis":
    st.header("📊 Batch Analysis")
    
    st.info("""
    This mode runs multiple simulations across different personal weight values
    to understand how misalignment correlates with shadow structure formation.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        n_orgs = st.number_input("Number of Organizations", 5, 50, 10)
        repeats = st.number_input("Repeats per Org", 5, 50, 10)
    with col2:
        steps = st.number_input("Steps per Simulation", 10, 500, 100)
        max_nodes = st.number_input("Max Nodes per Org", 5, 100, 30)
    
    if st.button("🚀 Run Batch Analysis", key="batch_run"):
        st.warning("⏳ This may take several minutes. Running in background...")
        st.info("Batch mode with real data analysis will be available when data files are present.")

# ============================================================================
# MODE: DATA EXPLORER
# ============================================================================
elif mode == "Data Explorer":
    st.header("🔍 Data Explorer")
    
    data_file = st.radio(
        "Select Data File",
        ["Master SP500 TMT", "Snapshot", "Simulation Results"],
        horizontal=True
    )
    
    if data_file == "Master SP500 TMT":
        st.subheader("SP500 Top Management Teams")
        try:
            df = pd.read_csv("master_SP500_TMT.csv", nrows=1000)
            
            col1, col2 = st.columns(2)
            with col1:
                company_filter = st.text_input("Filter by Company", "")
            with col2:
                year_filter = st.slider("Filter by Year", int(df["year"].min()), int(df["year"].max()), int(df["year"].min()))
            
            df_filtered = df.copy()
            if company_filter:
                df_filtered = df_filtered[df_filtered["company"].str.contains(company_filter, case=False, na=False)]
            df_filtered = df_filtered[df_filtered["year"] == year_filter]
            
            st.write(f"**Total Records:** {len(df_filtered)}")
            st.dataframe(df_filtered, use_container_width=True, height=400)
            
            # Summary statistics
            st.subheader("Summary Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Unique Companies", df_filtered["company"].nunique())
            with col2:
                st.metric("Unique Executives", df_filtered["full_name"].nunique())
            with col3:
                st.metric("Total Entries", len(df_filtered))
        
        except FileNotFoundError:
            st.error("master_SP500_TMT.csv not found")
    
    elif data_file == "Snapshot":
        st.subheader("Organization Snapshot")
        try:
            df = pd.read_csv("snapshot.csv", nrows=1000)
            
            company_filter = st.text_input("Filter by Company", "")
            df_filtered = df.copy()
            if company_filter:
                df_filtered = df_filtered[df_filtered["company"].str.contains(company_filter, case=False, na=False)]
            
            st.write(f"**Total Records:** {len(df_filtered)}")
            st.dataframe(df_filtered, use_container_width=True, height=400)
        
        except FileNotFoundError:
            st.error("snapshot.csv not found")
    
    elif data_file == "Simulation Results":
        st.subheader("Simulation Results")
        results_dir = Path("results")
        if results_dir.exists():
            result_files = list(results_dir.glob("*.csv"))
            if result_files:
                selected_file = st.selectbox("Select Result File", [f.name for f in result_files])
                df = pd.read_csv(results_dir / selected_file)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No result files found in results/ directory")
        else:
            st.info("No results/ directory found")

# ============================================================================
# MODE: CSV VISUALIZER
# ============================================================================
elif mode == "CSV Visualizer":
    st.header("📊 CSV Visualizer")
    
    st.markdown("""
    Browse, analyze, and visualize CSV files from the Simulations directory.
    Includes automatic chart generation, filtering, and statistical summaries.
    """)
    
    # Get all CSV files in Simulations directory
    csv_dir = Path(".")
    csv_files = sorted([f.name for f in csv_dir.glob("*.csv")])
    
    if not csv_files:
        st.warning("⚠️ No CSV files found in Simulations directory")
    else:
        st.subheader("Available CSV Files")
        selected_csv = st.selectbox(
            "Select a CSV file to visualize",
            csv_files,
            help="Choose a CSV file from the Simulations directory"
        )
        
        if selected_csv:
            csv_path = csv_dir / selected_csv
            
            try:
                # Load the CSV
                df = pd.read_csv(csv_path)
                
                # Display file info
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Rows", len(df))
                with col2:
                    st.metric("Columns", len(df.columns))
                with col3:
                    st.metric("File Size", f"{csv_path.stat().st_size / 1024:.1f} KB")
                with col4:
                    st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
                
                # Create tabs for different views
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📋 Data",
                    "📈 Visualizations",
                    "📊 Statistics",
                    "🔍 Analysis",
                    "💾 Export"
                ])
                
                with tab1:
                    st.subheader("Data Table")
                    
                    # Pagination controls
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        rows_per_page = st.number_input("Rows per page", 10, 1000, 50)
                    with col2:
                        page = st.number_input(
                            "Page",
                            1,
                            max(1, (len(df) + rows_per_page - 1) // rows_per_page),
                            1
                        )
                    with col3:
                        sort_col = st.selectbox("Sort by", ["None"] + df.columns.tolist())
                    
                    # Sort if needed
                    df_display = df.copy()
                    if sort_col != "None":
                        df_display = df_display.sort_values(sort_col, ascending=True)
                    
                    # Paginate
                    start_idx = (page - 1) * rows_per_page
                    end_idx = start_idx + rows_per_page
                    df_paginated = df_display.iloc[start_idx:end_idx]
                    
                    st.dataframe(df_paginated, use_container_width=True, height=500)
                    
                    # Column search
                    st.subheader("Search Columns")
                    search_col = st.selectbox("Search in column", df.columns.tolist())
                    search_term = st.text_input(f"Search term (in {search_col})")
                    if search_term:
                        df_search = df[df[search_col].astype(str).str.contains(search_term, case=False, na=False)]
                        st.write(f"Found {len(df_search)} matching rows")
                        st.dataframe(df_search, use_container_width=True, height=300)
                
                with tab2:
                    st.subheader("Automatic Visualizations")
                    
                    # Get numeric and categorical columns
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
                    
                    # Visualization type selector
                    viz_type = st.radio(
                        "Select visualization type",
                        ["Line Chart", "Bar Chart", "Histogram", "Scatter Plot", "Box Plot", "Heatmap"],
                        horizontal=True
                    )
                    
                    if viz_type == "Line Chart":
                        if numeric_cols:
                            x_col = st.selectbox("X-axis", df.columns.tolist(), key="line_x")
                            y_cols = st.multiselect(
                                "Y-axis (numeric columns)",
                                numeric_cols,
                                default=[numeric_cols[0]] if numeric_cols else None,
                                key="line_y"
                            )
                            if y_cols:
                                fig = go.Figure()
                                for y_col in y_cols:
                                    fig.add_trace(go.Scatter(
                                        x=df[x_col],
                                        y=df[y_col],
                                        mode='lines+markers',
                                        name=y_col
                                    ))
                                fig.update_layout(
                                    title=f"{selected_csv} - Line Chart",
                                    xaxis_title=x_col,
                                    yaxis_title="Values",
                                    hovermode='x unified',
                                    height=500
                                )
                                st.plotly_chart(fig, use_container_width=True)
                    
                    elif viz_type == "Bar Chart":
                        if cat_cols and numeric_cols:
                            x_col = st.selectbox("Category (X-axis)", cat_cols, key="bar_x")
                            y_col = st.selectbox("Value (Y-axis)", numeric_cols, key="bar_y")
                            
                            # Group and aggregate
                            grouped = df.groupby(x_col)[y_col].sum().reset_index()
                            fig = px.bar(grouped, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                            fig.update_layout(height=500)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    elif viz_type == "Histogram":
                        if numeric_cols:
                            col = st.selectbox("Select column", numeric_cols, key="hist_col")
                            bins = st.slider("Number of bins", 10, 100, 30, key="hist_bins")
                            
                            fig = px.histogram(df, x=col, nbins=bins, title=f"Distribution of {col}")
                            fig.update_layout(height=500)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    elif viz_type == "Scatter Plot":
                        if len(numeric_cols) >= 2:
                            x_col = st.selectbox("X-axis", numeric_cols, key="scatter_x")
                            y_col = st.selectbox(
                                "Y-axis",
                                [c for c in numeric_cols if c != x_col],
                                key="scatter_y"
                            )
                            color_col = st.selectbox(
                                "Color (optional)",
                                ["None"] + df.columns.tolist(),
                                key="scatter_color"
                            )
                            
                            if color_col == "None":
                                fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
                            else:
                                fig = px.scatter(
                                    df, x=x_col, y=y_col,
                                    color=color_col,
                                    title=f"{x_col} vs {y_col} (colored by {color_col})"
                                )
                            fig.update_layout(height=500)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    elif viz_type == "Box Plot":
                        if numeric_cols and cat_cols:
                            cat_col = st.selectbox("Category", cat_cols, key="box_cat")
                            num_col = st.selectbox("Value", numeric_cols, key="box_num")
                            
                            fig = px.box(df, x=cat_col, y=num_col, title=f"{num_col} by {cat_col}")
                            fig.update_layout(height=500)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    elif viz_type == "Heatmap":
                        if len(numeric_cols) >= 2:
                            # Select columns for correlation
                            cols_to_correlate = st.multiselect(
                                "Select columns for correlation",
                                numeric_cols,
                                default=numeric_cols[:min(5, len(numeric_cols))],
                                key="heat_cols"
                            )
                            
                            if cols_to_correlate:
                                corr_matrix = df[cols_to_correlate].corr()
                                fig = px.imshow(
                                    corr_matrix,
                                    title="Correlation Matrix",
                                    color_continuous_scale="RdBu",
                                    zmin=-1, zmax=1
                                )
                                fig.update_layout(height=500)
                                st.plotly_chart(fig, use_container_width=True)
                
                with tab3:
                    st.subheader("Statistical Summary")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Numeric Columns Summary**")
                        numeric_summary = df.describe()
                        st.dataframe(numeric_summary, use_container_width=True)
                    
                    with col2:
                        st.write("**Data Types**")
                        dtype_summary = pd.DataFrame({
                            'Column': df.columns,
                            'Type': df.dtypes,
                            'Non-Null': df.count(),
                            'Null': df.isnull().sum()
                        })
                        st.dataframe(dtype_summary, use_container_width=True)
                    
                    st.write("**Missing Values**")
                    missing = pd.DataFrame({
                        'Column': df.columns,
                        'Missing Count': df.isnull().sum(),
                        'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
                    })
                    missing = missing[missing['Missing Count'] > 0]
                    if len(missing) > 0:
                        st.dataframe(missing, use_container_width=True)
                    else:
                        st.success("✓ No missing values")
                    
                    # Numeric column statistics
                    if numeric_cols:
                        st.write("**Numeric Column Details**")
                        for col in numeric_cols:
                            with st.expander(f"📊 {col}"):
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Mean", f"{df[col].mean():.3f}")
                                    st.metric("Median", f"{df[col].median():.3f}")
                                with col2:
                                    st.metric("Std Dev", f"{df[col].std():.3f}")
                                    st.metric("Min", f"{df[col].min():.3f}")
                                with col3:
                                    st.metric("Max", f"{df[col].max():.3f}")
                                    st.metric("Q1", f"{df[col].quantile(0.25):.3f}")
                
                with tab4:
                    st.subheader("Advanced Analysis")
                    
                    analysis_type = st.selectbox(
                        "Select analysis",
                        ["Correlation Analysis", "Top Values", "Distribution Comparison", "Time Series Trend"]
                    )
                    
                    if analysis_type == "Correlation Analysis":
                        if len(numeric_cols) >= 2:
                            st.write("Correlation between numeric columns:")
                            corr = df[numeric_cols].corr()
                            
                            # Find strong correlations
                            strong_corr = []
                            for i in range(len(corr.columns)):
                                for j in range(i+1, len(corr.columns)):
                                    if abs(corr.iloc[i, j]) > 0.7:
                                        strong_corr.append({
                                            'Column 1': corr.columns[i],
                                            'Column 2': corr.columns[j],
                                            'Correlation': f"{corr.iloc[i, j]:.3f}"
                                        })
                            
                            if strong_corr:
                                st.write("**Strong Correlations (|r| > 0.7)**")
                                st.dataframe(pd.DataFrame(strong_corr), use_container_width=True)
                            else:
                                st.info("No strong correlations found")
                    
                    elif analysis_type == "Top Values":
                        if numeric_cols or cat_cols:
                            col_to_analyze = st.selectbox("Select column", df.columns.tolist())
                            n_top = st.slider("Number of top values", 5, 50, 10)
                            
                            if df[col_to_analyze].dtype == 'object':
                                top_vals = df[col_to_analyze].value_counts().head(n_top)
                                fig = px.bar(
                                    x=top_vals.index,
                                    y=top_vals.values,
                                    title=f"Top {n_top} values in {col_to_analyze}",
                                    labels={'x': col_to_analyze, 'y': 'Count'}
                                )
                            else:
                                top_vals = df.nlargest(n_top, col_to_analyze)[[col_to_analyze]]
                                fig = px.bar(
                                    top_vals.reset_index(),
                                    x='index',
                                    y=col_to_analyze,
                                    title=f"Top {n_top} values in {col_to_analyze}"
                                )
                            
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    elif analysis_type == "Distribution Comparison":
                        if len(numeric_cols) >= 1:
                            col1 = st.selectbox("First column", numeric_cols, key="dist_col1")
                            col2 = st.selectbox(
                                "Second column",
                                [c for c in numeric_cols if c != col1],
                                key="dist_col2"
                            )
                            
                            fig = go.Figure()
                            fig.add_trace(go.Histogram(
                                x=df[col1],
                                name=col1,
                                opacity=0.7
                            ))
                            fig.add_trace(go.Histogram(
                                x=df[col2],
                                name=col2,
                                opacity=0.7
                            ))
                            fig.update_layout(
                                barmode='overlay',
                                title=f"Distribution Comparison: {col1} vs {col2}",
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    elif analysis_type == "Time Series Trend":
                        if numeric_cols:
                            time_col = st.selectbox("Time/X-axis column", df.columns.tolist())
                            value_col = st.selectbox("Value/Y-axis column", numeric_cols)
                            
                            # Try to sort by time if possible
                            df_sorted = df.sort_values(time_col, ascending=True)
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=df_sorted[time_col],
                                y=df_sorted[value_col],
                                mode='lines',
                                name=value_col
                            ))
                            fig.update_layout(
                                title=f"Trend: {value_col} over {time_col}",
                                xaxis_title=time_col,
                                yaxis_title=value_col,
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                
                with tab5:
                    st.subheader("Export Data")
                    
                    # CSV export
                    csv_export = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download as CSV",
                        data=csv_export,
                        file_name=f"{selected_csv.replace('.csv', '')}_export.csv",
                        mime="text/csv"
                    )
                    
                    # Excel export
                    try:
                        excel_buffer = StringIO()
                        with pd.ExcelWriter('temp_export.xlsx') as writer:
                            df.to_excel(writer, index=False)
                        with open('temp_export.xlsx', 'rb') as f:
                            excel_data = f.read()
                        st.download_button(
                            label="📊 Download as Excel",
                            data=excel_data,
                            file_name=f"{selected_csv.replace('.csv', '')}_export.xlsx",
                            mime="application/vnd.ms-excel"
                        )
                    except:
                        st.info("Excel export requires openpyxl library")
                    
                    # JSON export
                    json_export = df.to_json(orient='records', indent=2)
                    st.download_button(
                        label="📋 Download as JSON",
                        data=json_export,
                        file_name=f"{selected_csv.replace('.csv', '')}_export.json",
                        mime="application/json"
                    )
                    
                    st.write("**Summary**")
                    st.info(f"Export includes {len(df)} rows and {len(df.columns)} columns")
            
            except Exception as e:
                st.error(f"Error loading CSV: {str(e)}")

# ============================================================================
# Footer
# ============================================================================
st.markdown("---")
st.markdown("""
**About:** This simulator models how organizational misalignment leads to shadow structures.
Agents compete between organizational goals, personal interests, and global safety.
""")
