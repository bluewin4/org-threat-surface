# CSV Visualizer Guide

## Overview

The **CSV Visualizer** is a new mode in the Organization Threat Surface Simulator that allows you to explore, analyze, and visualize any CSV files in the `Simulations/` directory.

## Access

1. Run the UI: `./start_ui.sh`
2. Go to sidebar
3. Select **"CSV Visualizer"** from the Mode selector

## Features

### 📊 Available CSV Files

Automatically detects all CSV files in the Simulations directory:
- `master_SP500_TMT.csv` - SP500 Top Management Teams (50MB+)
- `snapshot.csv` - Organization snapshots
- `codebook.csv` - Data dictionary
- Any exported simulation results

### 📋 Data Tab

**View & Navigate Data:**
- Paginated table view (adjustable rows per page)
- Sort by any column
- Search within specific columns
- Display file info (rows, columns, file size, memory usage)

### 📈 Visualizations Tab

**6 Chart Types:**

1. **Line Chart**
   - Multi-line support
   - Great for time series or trends
   - Multiple Y-axes on same plot

2. **Bar Chart**
   - Category aggregation
   - Sum/count operations
   - Perfect for comparisons

3. **Histogram**
   - Distribution analysis
   - Adjustable bin count
   - Shows frequency patterns

4. **Scatter Plot**
   - Two-variable relationships
   - Optional color coding
   - Correlation visualization

5. **Box Plot**
   - Category-wise distributions
   - Outlier detection
   - Quartile visualization

6. **Heatmap**
   - Correlation matrices
   - Select up to N columns
   - Color-coded strength (-1 to +1)

### 📊 Statistics Tab

**Automatic Statistics:**
- Numeric summary (mean, median, std dev, min, max, percentiles)
- Data type overview
- Missing value detection
- Per-column detailed metrics

### 🔍 Analysis Tab

**Advanced Analytics:**

1. **Correlation Analysis**
   - Find strong correlations (|r| > 0.7)
   - Pairwise comparison table
   - Identify relationships

2. **Top Values**
   - Most frequent categories
   - Highest numeric values
   - Adjustable count (5-50)

3. **Distribution Comparison**
   - Compare two columns side-by-side
   - Overlapping histograms
   - Visual difference assessment

4. **Time Series Trend**
   - Plot values over time/X-axis
   - Automatic sorting
   - Trend identification

### 💾 Export Tab

**Download in Multiple Formats:**
- **CSV** - Compatible with Excel, Python, R
- **Excel** - Native spreadsheet format
- **JSON** - Machine-readable, nested data

## Usage Examples

### Example 1: Explore Executive Data

1. Select `master_SP500_TMT.csv`
2. Go to **Statistics** tab
3. See overview of executive data
4. Check missing values
5. Go to **Visualizations** → Bar Chart
6. Plot "role" vs "year" to see distribution

### Example 2: Analyze Organization Structure

1. Select `snapshot.csv`
2. Go to **Data** tab
3. Search for specific companies
4. Go to **Visualizations** → Scatter Plot
5. Plot organizational metrics against each other
6. Use color coding to reveal patterns

### Example 3: Correlation Deep Dive

1. Select any simulation results CSV
2. Go to **Analysis** → Correlation Analysis
3. Identify strongly correlated variables
4. Go to **Visualizations** → Scatter Plot
5. Visualize top correlations

### Example 4: Export Filtered Data

1. Select desired CSV
2. Go to **Data** tab
3. Search/filter to subset
4. Copy filtered data from table
5. Go to **Export** tab
6. Download in your preferred format

## Tips & Tricks

### Performance
- Large files (>10MB) may take a moment to load
- Use pagination to avoid rendering huge tables
- Filter/search to reduce data size before charting

### Data Quality
- **Missing Values** tab shows data completeness
- Watch for `NaN`, `None`, or empty cells
- Some visualizations require complete data

### Chart Selection
- **Time data?** → Line Chart or Time Series Trend
- **Categories?** → Bar Chart or Box Plot
- **Relationships?** → Scatter Plot or Correlation
- **Distributions?** → Histogram
- **All relationships?** → Heatmap

### Export Strategy
- CSV → Excel compatibility, universal format
- Excel → Better formatting, multiple sheets ready
- JSON → API integration, nested data support

## Common Tasks

### Find Outliers
1. Statistics tab → Review Min/Max
2. Box Plot visualization
3. Manual inspection in Data tab

### Compare Groups
1. Select grouping column
2. Use Box Plot to see distributions per group
3. Use Bar Chart to aggregate values

### Check Data Quality
1. Statistics tab → Data Types
2. Look for "object" where you expect numbers
3. Review Missing Values section
4. Fix in source data if needed

### Build Custom Report
1. Visualize key metrics (3-5 charts)
2. Screenshot or save each
3. Export data with Export tab
4. Combine in external document

## Limitations

| Limitation | Workaround |
|-----------|-----------|
| CSV too large (>1GB) | Load in chunks externally, or export subset |
| Date columns not recognized | Convert to appropriate format in source data |
| Too many categories | Filter/aggregate data first |
| Slow performance | Use pagination, reduce rows, or pre-process |
| Excel export fails | Ensure openpyxl installed, use CSV instead |

## Troubleshooting

**Q: "No CSV files found"**
- Check CSV files are in `/Users/mp/org-threat-surface/Simulations/`
- Make sure they have `.csv` extension
- Run `ls *.csv` in Simulations directory

**Q: Chart won't display**
- Check if selected columns are correct type (numeric for Y-axis)
- Try simpler visualization (Line Chart first)
- Verify data contains expected values

**Q: Slow loading**
- Using very large CSV? Consider pre-filtering
- Try pagination instead of full table view
- Reduce number of rows displayed

**Q: Search not working**
- Column might have null values
- Try partial text matches
- Check case sensitivity (searches ignore case)

**Q: Excel export failed**
- Install openpyxl: `pip install openpyxl`
- Use CSV export as fallback
- JSON export always works

## Feature Requests

Want more visualizations? Open an issue with:
- Chart type name
- Use case (what problem does it solve?)
- Example data structure

## Integration

### With Simulation Results
1. Run simulation → exports CSV
2. Switch to **CSV Visualizer**
3. Select your results CSV
4. Analyze, visualize, export

### With External Data
1. Place CSV in `Simulations/` directory
2. Refresh/restart app
3. Select from dropdown
4. Visualize immediately

## Related Modes

- **Data Explorer** - Browse data sources (limited)
- **CSV Visualizer** - Full analysis & visualization (current)
- **Quick Demo** - Simulate + visualize results
- **Custom Simulation** - Generate new data for visualization

---

**Version:** 1.0  
**Added:** November 2025  
**Supported Formats:** CSV  
**Charts:** 6 types  
**Export Formats:** CSV, Excel, JSON
