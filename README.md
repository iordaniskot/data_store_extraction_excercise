# Clustering & Anomaly Detection on 2D Data

A Python implementation of **k-means clustering** and **outlier detection** for 2D datasets.

> **Course:** Αποθήκες και Εξόρυξη Δεδομένων 2025–2026  


---

## 📋 Table of Contents

- [Features](#-features)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Input File Formats](#-input-file-formats)
- [Configuration](#-configuration)
- [Output](#-output)
- [Example](#-example)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)

---

## ✨ Features

- **K-Means Clustering** with multiple initialization methods (random, k-means++, uniform)
- **Outlier Detection** using 4 different methods:
  - Distance-based (mean + k×std)
  - IQR-based
  - Percentile-based
  - Small cluster detection
- **Automatic file conversion** from Excel, JSON, TSV, and TXT formats
- **Data cleaning** with flexible missing value handling
- **Optional normalization** with preservation of original coordinates
- **Execution timing** for performance analysis
- **Reproducible results** with random seed support

---

## 📦 Requirements

- **Python 3.7+**
- No external dependencies required for basic functionality

### Optional Dependencies

| Package | Required For |
|---------|--------------|
| `pandas` | Excel file conversion (.xlsx, .xls) |
| `openpyxl` | Excel file conversion (.xlsx) |

---

## 🚀 Installation

### 1. Clone or Download the Repository

```bash
git clone <repository-url>
cd data_store_extraction_excercise
```

### 2. (Optional) Create a Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### 3. (Optional) Install Dependencies for Excel Support

```bash
pip install pandas openpyxl
```

---

## ⚡ Quick Start

Run the program with a CSV file:

```bash
python main.py data/demo_dataset.csv
```

That's it! The program will:
1. Load and clean the data
2. Perform k-means clustering (k=5)
3. Detect outliers using distance-based method
4. Print results with execution time

---

## 📖 Usage

### Basic Syntax

```bash
python main.py <path_to_data_file>
```

### Examples

```bash
# Using CSV file
python main.py data/demo_dataset.csv

# Using text file (auto-converted)
python main.py data/data202526a_corrupted.txt

# Using absolute path
python main.py /path/to/your/data.csv

# Using relative path
python main.py ../other_folder/data.csv
```

### Command-Line Help

```bash
python main.py --help
```

Output:
```
usage: main.py [-h] filename

Clustering & Anomaly Detection on 2D CSV data

positional arguments:
  filename    Path to the CSV file containing 2D data points

optional arguments:
  -h, --help  show this help message and exit

Example: python main.py data/data202526a.csv
```

---

## 📁 Input File Formats

### Supported Formats

| Format | Extensions | Notes |
|--------|------------|-------|
| CSV | `.csv` | Native format (recommended) |
| Text | `.txt` | Auto-detects delimiter |
| TSV | `.tsv` | Tab-separated values |
| Excel | `.xlsx`, `.xls` | Requires pandas |
| JSON | `.json` | Array of objects or 2D array |

### CSV Format Requirements

- **Two columns** representing X and Y coordinates
- **No header row** (all rows treated as data)
- Values can be integers or decimals

**Example CSV:**
```csv
1.5,2.3
3.1,4.2
-0.5,7.8
10.2,5.5
```

### Handling Corrupted Data

The program automatically handles:
- ✅ Missing values (empty cells)
- ✅ Malformed rows (non-numeric values)
- ✅ Rows with fewer than 2 columns
- ✅ Infinity and NaN values

Removed rows are logged with reasons.

---

## ⚙️ Configuration

### Modifying Parameters

Edit `main.py` to change default settings:

```python
# Number of clusters
K = 5

# Data cleaning options
data_result = utils.clean_and_validate_data(
    raw_data, 
    handle_missing='remove',  # Options: 'remove', 'mean', 'median', 'zero'
    normalize=False           # Set to True for normalization
)

# K-means options
kmeans_result = utils.kmeans(
    cleaned_data,
    k=K,
    max_iterations=100,
    tolerance=1e-6,
    init_method='kmeans++',   # Options: 'random', 'kmeans++', 'uniform'
    random_seed=42            # For reproducibility (or None for random)
)

# Outlier detection options
outlier_result = utils.detect_outliers(
    cleaned_data,
    kmeans_result,
    original_data=original_data,
    method='distance',        # Options: 'distance', 'iqr', 'percentile', 'small_cluster'
    threshold=2.0             # Depends on method
)
```

### Outlier Detection Methods

| Method | Threshold Meaning | Default |
|--------|-------------------|---------|
| `distance` | Number of standard deviations | 2.0 |
| `iqr` | IQR multiplier | 1.5 |
| `percentile` | Percentile (0-100) | 95 (top 5%) |
| `small_cluster` | Minimum cluster size | 5% of data |

---

## 📤 Output

The program outputs:

1. **Data Loading Summary**
   - Number of rows loaded
   - File path used

2. **Data Cleaning Summary**
   - Total rows, valid rows, removed rows
   - Removal rate percentage
   - Details of first 10 removed rows

3. **K-Means Results**
   - Configuration (k, initialization method, tolerance)
   - Convergence status
   - Number of iterations
   - Inertia (SSE)
   - Cluster sizes

4. **Outlier Detection Results**
   - Detection method and threshold
   - Number of outliers found
   - Percentage of data flagged as outliers

5. **Outlier List**
   - Index, X, Y coordinates (original), distance, cluster

6. **Execution Time**
   - Total runtime in seconds and milliseconds

---

## 📝 Example

### Running with Demo Dataset

```bash
python main.py data/demo_dataset.csv
```

### Sample Output

```
Loaded 1000 rows from: data/demo_dataset.csv

============================================================
Data Cleaning Summary:
  Total rows: 1000
  Valid rows: 985
  Removed rows: 15 (1.50%)
============================================================

Removed rows details:
  Row 42: missing_value
  Row 156: malformed_row: could not convert string to float: 'N/A'
  ...

============================================================
K-Means Clustering
  k = 5
  Initialization: kmeans++
  Max iterations: 100
  Tolerance: 1e-06
============================================================

Results:
  Converged: True
  Iterations: 8
  Inertia (SSE): 1234.5678

Cluster sizes:
  Cluster 0: 198 points
  Cluster 1: 205 points
  Cluster 2: 189 points
  Cluster 3: 201 points
  Cluster 4: 192 points
============================================================

============================================================
Outlier Detection Results
  Method: distance
  Threshold: 2.3456
  Outliers found: 47 (4.77%)
============================================================

============================================================
DETECTED OUTLIERS (Original Coordinates)
============================================================
Index          X               Y     Distance  Cluster
------------------------------------------------------------
234        15.234567       28.891234       3.2145        2
567         8.123456      -12.456789       2.8976        0
...
============================================================
Total outliers: 47
============================================================

Program completed successfully.
  Total data points: 985
  Clusters: 5
  Outliers detected: 47

============================================================
Execution time: 0.1234 seconds (123.40 ms)
============================================================
```

---

## 📂 Project Structure

```
data_store_extraction_excercise/
├── main.py                 # Main entry point
├── utils.py                # All utility functions and algorithms
├── README.md               # This file
├── data/                   # Data files
│   ├── demo_dataset.csv
│   ├── data202526a_corrupted.txt
│   ├── data202526b_corrupted.txt
│   └── ...
└── __pycache__/            # Python cache (auto-generated)
```

---

## 🔧 Troubleshooting

### Common Issues

#### "File not found" Error

```
Error: File 'data.csv' does not exist.
```

**Solution:** Check the file path. Use absolute path or ensure you're running from the correct directory.

```bash
# Check current directory
pwd

# List files in data folder
ls data/
```

#### "File not readable" Error

```
Error: File 'data.csv' is not readable.
```

**Solution:** Check file permissions.

```bash
chmod +r data.csv
```

#### Excel Conversion Error

```
Error: pandas is required for Excel conversion.
```

**Solution:** Install pandas and openpyxl.

```bash
pip install pandas openpyxl
```

#### No Outliers Detected

If the program finds 0 outliers:
- Try lowering the threshold (e.g., 1.5 instead of 2.0)
- Try a different detection method
- Check if your data actually has anomalies

#### All Points Marked as Outliers

If too many points are flagged:
- Increase the threshold
- Check if k value is appropriate for your data
- Try different values of k (e.g., 3, 4, 5, 6)



## 🤝 Authors

- **Student AEM:** 4373
