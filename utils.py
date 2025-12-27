import time
import functools


def execution_timer(func):
    """Decorator to measure and print execution time of a function.
    
    Args:
        func: The function to be timed.
    
    Returns:
        The wrapped function that prints execution time.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\n{'='*60}")
        print(f"Execution time: {execution_time:.4f} seconds ({execution_time*1000:.2f} ms)")
        print(f"{'='*60}")
        return result
    return wrapper


def csv_reader(file_path):
    """Reads a CSV file and returns its content as a list of dictionaries.
    
    All rows (including the first) are treated as data points.
    Each row is expected to have exactly 2 columns representing x and y coordinates.

    Args:
        file_path (str): The path to the CSV file.
    Returns:
        list: A list of dictionaries with keys 'x' and 'y' representing 2D data points.
    """
    import csv

    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        data = []
        for row in reader:
            if len(row) >= 2:  # Ensure at least 2 columns exist
                data.append({'x': row[0], 'y': row[1]})
        return data


def clean_and_validate_data(raw_data, handle_missing='remove', normalize=False):
    """Validates, cleans, and optionally normalizes 2D data points.
    
    This function:
    - Validates numeric input
    - Handles missing values
    - Handles malformed rows
    - Stores original data
    - Optionally normalizes data
    
    Args:
        raw_data (list): List of dictionaries with 'x' and 'y' keys (from csv_reader).
        handle_missing (str): How to handle missing values:
            - 'remove': Remove rows with missing values (default)
            - 'mean': Replace missing values with column mean
            - 'median': Replace missing values with column median
            - 'zero': Replace missing values with 0
        normalize (bool): Whether to normalize data to [0, 1] range.
    
    Returns:
        dict: Dictionary containing:
            - 'cleaned_data': List of cleaned/validated data points
            - 'original_data': Copy of original data (before normalization)
            - 'normalization_params': Min/max values if normalized, else None
            - 'removed_rows': List of removed row indices and reasons
            - 'stats': Statistics about cleaning process
    """
    import copy
    import sys
    
    cleaned_data = []
    removed_rows = []
    original_indices = []
    
    # Step 1: Validate and clean data
    for idx, row in enumerate(raw_data):
        try:
            x_str = row.get('x', '').strip()
            y_str = row.get('y', '').strip()
            
            # Check for missing values
            if not x_str or not y_str:
                if handle_missing == 'remove':
                    removed_rows.append({
                        'index': idx,
                        'reason': 'missing_value',
                        'data': row
                    })
                    continue
                elif handle_missing in ['mean', 'median', 'zero']:
                    # Will handle after first pass
                    x_val = None if not x_str else float(x_str)
                    y_val = None if not y_str else float(y_str)
                else:
                    removed_rows.append({
                        'index': idx,
                        'reason': 'missing_value',
                        'data': row
                    })
                    continue
            else:
                # Try to convert to float
                x_val = float(x_str)
                y_val = float(y_str)
            
            # Check for invalid numeric values (inf, nan)
            if x_val is not None and y_val is not None:
                if not (float('-inf') < x_val < float('inf')) or \
                   not (float('-inf') < y_val < float('inf')):
                    removed_rows.append({
                        'index': idx,
                        'reason': 'invalid_numeric_value',
                        'data': row
                    })
                    continue
            
            cleaned_data.append({'x': x_val, 'y': y_val})
            original_indices.append(idx)
            
        except (ValueError, TypeError) as e:
            # Malformed row - cannot convert to float
            removed_rows.append({
                'index': idx,
                'reason': f'malformed_row: {str(e)}',
                'data': row
            })
            continue
    
    # Step 2: Handle missing values (if not 'remove')
    if handle_missing in ['mean', 'median', 'zero']:
        # Collect non-None values
        x_values = [p['x'] for p in cleaned_data if p['x'] is not None]
        y_values = [p['y'] for p in cleaned_data if p['y'] is not None]
        
        if handle_missing == 'mean':
            x_fill = sum(x_values) / len(x_values) if x_values else 0
            y_fill = sum(y_values) / len(y_values) if y_values else 0
        elif handle_missing == 'median':
            x_sorted = sorted(x_values)
            y_sorted = sorted(y_values)
            x_fill = x_sorted[len(x_sorted)//2] if x_sorted else 0
            y_fill = y_sorted[len(y_sorted)//2] if y_sorted else 0
        else:  # 'zero'
            x_fill = 0
            y_fill = 0
        
        # Fill missing values
        for point in cleaned_data:
            if point['x'] is None:
                point['x'] = x_fill
            if point['y'] is None:
                point['y'] = y_fill
    
    # Step 3: Store original data (deep copy before normalization)
    original_data = copy.deepcopy(cleaned_data)
    
    # Step 4: Optional normalization
    normalization_params = None
    if normalize and cleaned_data:
        x_values = [p['x'] for p in cleaned_data]
        y_values = [p['y'] for p in cleaned_data]
        
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        
        normalization_params = {
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max
        }
        
        # Normalize to [0, 1]
        x_range = x_max - x_min if x_max != x_min else 1
        y_range = y_max - y_min if y_max != y_min else 1
        
        for point in cleaned_data:
            point['x'] = (point['x'] - x_min) / x_range
            point['y'] = (point['y'] - y_min) / y_range
    
    # Step 5: Generate statistics
    stats = {
        'total_rows': len(raw_data),
        'valid_rows': len(cleaned_data),
        'removed_rows': len(removed_rows),
        'removal_rate': len(removed_rows) / len(raw_data) * 100 if raw_data else 0,
        'normalized': normalize
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Data Cleaning Summary:")
    print(f"  Total rows: {stats['total_rows']}")
    print(f"  Valid rows: {stats['valid_rows']}")
    print(f"  Removed rows: {stats['removed_rows']} ({stats['removal_rate']:.2f}%)")
    if normalize:
        print(f"  Normalization: Applied (range [0, 1])")
    print(f"{'='*60}\n")
    
    if removed_rows:
        print(f"Removed rows details:")
        for removal in removed_rows[:10]:  # Show first 10
            print(f"  Row {removal['index']}: {removal['reason']}")
        if len(removed_rows) > 10:
            print(f"  ... and {len(removed_rows) - 10} more")
        print()
    
    return {
        'cleaned_data': cleaned_data,
        'original_data': original_data,
        'normalization_params': normalization_params,
        'removed_rows': removed_rows,
        'original_indices': original_indices,
        'stats': stats
    }


def denormalize_point(point, normalization_params):
    """Converts a normalized point back to original coordinates.
    
    Args:
        point (dict): Normalized point with 'x' and 'y' keys.
        normalization_params (dict): Parameters from clean_and_validate_data.
    
    Returns:
        dict: Point in original coordinate space.
    """
    if normalization_params is None:
        return point
    
    x_min = normalization_params['x_min']
    x_max = normalization_params['x_max']
    y_min = normalization_params['y_min']
    y_max = normalization_params['y_max']
    
    x_range = x_max - x_min if x_max != x_min else 1
    y_range = y_max - y_min if y_max != y_min else 1
    
    return {
        'x': point['x'] * x_range + x_min,
        'y': point['y'] * y_range + y_min
    }


# =============================================================================
# K-MEANS CLUSTERING IMPLEMENTATION
# =============================================================================

def euclidean_distance(point1, point2):
    """Computes the Euclidean distance between two 2D points.
    
    Args:
        point1 (dict): First point with 'x' and 'y' keys.
        point2 (dict): Second point with 'x' and 'y' keys.
    
    Returns:
        float: Euclidean distance between the two points.
    """
    import math
    return math.sqrt((point1['x'] - point2['x'])**2 + (point1['y'] - point2['y'])**2)


def initialize_centroids(data, k, method='random'):
    """Initializes k centroids for k-means clustering.
    
    Args:
        data (list): List of data points with 'x' and 'y' keys.
        k (int): Number of clusters.
        method (str): Initialization method:
            - 'random': Randomly select k points from data
            - 'kmeans++': K-means++ initialization for better convergence
            - 'uniform': Uniformly distributed across data range
    
    Returns:
        list: List of k centroid points.
    """
    import random
    
    if k > len(data):
        raise ValueError(f"k ({k}) cannot be greater than number of data points ({len(data)})")
    
    if method == 'random':
        # Randomly select k unique points as initial centroids
        selected = random.sample(data, k)
        return [{'x': p['x'], 'y': p['y']} for p in selected]
    
    elif method == 'kmeans++':
        # K-means++ initialization: select centroids with probability
        # proportional to distance squared from nearest existing centroid
        centroids = []
        
        # First centroid: random
        first = random.choice(data)
        centroids.append({'x': first['x'], 'y': first['y']})
        
        for _ in range(1, k):
            distances = []
            for point in data:
                # Find minimum distance to existing centroids
                min_dist = min(euclidean_distance(point, c) for c in centroids)
                distances.append(min_dist ** 2)
            
            # Select next centroid with probability proportional to distance²
            total_dist = sum(distances)
            if total_dist == 0:
                # All points are at centroid positions, select random
                next_idx = random.randint(0, len(data) - 1)
            else:
                probabilities = [d / total_dist for d in distances]
                next_idx = random.choices(range(len(data)), weights=probabilities, k=1)[0]
            
            centroids.append({'x': data[next_idx]['x'], 'y': data[next_idx]['y']})
        
        return centroids
    
    elif method == 'uniform':
        # Uniformly distribute centroids across data range
        x_values = [p['x'] for p in data]
        y_values = [p['y'] for p in data]
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        
        centroids = []
        for i in range(k):
            x = x_min + (x_max - x_min) * (i + 0.5) / k
            y = y_min + (y_max - y_min) * random.random()
            centroids.append({'x': x, 'y': y})
        
        return centroids
    
    else:
        raise ValueError(f"Unknown initialization method: {method}")


def assign_clusters(data, centroids):
    """Assigns each data point to the nearest centroid.
    
    Args:
        data (list): List of data points.
        centroids (list): List of centroid points.
    
    Returns:
        list: List of cluster assignments (indices into centroids list).
    """
    assignments = []
    for point in data:
        distances = [euclidean_distance(point, c) for c in centroids]
        nearest_cluster = distances.index(min(distances))
        assignments.append(nearest_cluster)
    return assignments


def compute_centroids(data, assignments, k):
    """Computes new centroids as the mean of assigned points.
    
    Args:
        data (list): List of data points.
        assignments (list): Cluster assignments for each point.
        k (int): Number of clusters.
    
    Returns:
        list: List of new centroid points.
    """
    new_centroids = []
    
    for cluster_id in range(k):
        # Get all points assigned to this cluster
        cluster_points = [data[i] for i in range(len(data)) if assignments[i] == cluster_id]
        
        if cluster_points:
            # Compute mean
            mean_x = sum(p['x'] for p in cluster_points) / len(cluster_points)
            mean_y = sum(p['y'] for p in cluster_points) / len(cluster_points)
            new_centroids.append({'x': mean_x, 'y': mean_y})
        else:
            # Empty cluster: keep old centroid or reinitialize randomly
            # Here we keep a placeholder that will be handled
            new_centroids.append({'x': 0, 'y': 0})
    
    return new_centroids


def has_converged(old_centroids, new_centroids, tolerance=1e-6):
    """Checks if centroids have converged.
    
    Args:
        old_centroids (list): Previous centroid positions.
        new_centroids (list): New centroid positions.
        tolerance (float): Maximum allowed movement for convergence.
    
    Returns:
        bool: True if converged, False otherwise.
    """
    for old, new in zip(old_centroids, new_centroids):
        if euclidean_distance(old, new) > tolerance:
            return False
    return True


def kmeans(data, k, max_iterations=100, tolerance=1e-6, init_method='kmeans++', random_seed=None):
    """Performs k-means clustering on 2D data.
    
    Algorithm:
    1. Initialize k centroids using specified method
    2. Assign each point to the nearest centroid
    3. Recompute centroids as mean of assigned points
    4. Repeat until convergence or max iterations reached
    
    Args:
        data (list): List of data points with 'x' and 'y' keys.
        k (int): Number of clusters.
        max_iterations (int): Maximum number of iterations (default: 100).
        tolerance (float): Convergence tolerance for centroid movement (default: 1e-6).
        init_method (str): Centroid initialization method: 'random', 'kmeans++', 'uniform'.
        random_seed (int): Random seed for reproducibility (default: None).
    
    Returns:
        dict: Dictionary containing:
            - 'centroids': Final centroid positions
            - 'assignments': Cluster assignment for each data point
            - 'clusters': List of lists, each containing points in that cluster
            - 'iterations': Number of iterations performed
            - 'converged': Whether algorithm converged
            - 'inertia': Sum of squared distances to nearest centroid (SSE)
    """
    import random
    
    if random_seed is not None:
        random.seed(random_seed)
    
    if len(data) == 0:
        raise ValueError("Cannot perform k-means on empty dataset")
    
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    
    # Step 1: Initialize centroids
    centroids = initialize_centroids(data, k, method=init_method)
    
    iterations = 0
    converged = False
    
    print(f"\n{'='*60}")
    print(f"K-Means Clustering")
    print(f"  k = {k}")
    print(f"  Initialization: {init_method}")
    print(f"  Max iterations: {max_iterations}")
    print(f"  Tolerance: {tolerance}")
    print(f"{'='*60}")
    
    # Steps 2-4: Iterate until convergence
    for iteration in range(max_iterations):
        iterations = iteration + 1
        
        # Assign points to nearest centroid
        assignments = assign_clusters(data, centroids)
        
        # Compute new centroids
        new_centroids = compute_centroids(data, assignments, k)
        
        # Check for convergence
        if has_converged(centroids, new_centroids, tolerance):
            converged = True
            centroids = new_centroids
            break
        
        centroids = new_centroids
    
    # Final assignment
    assignments = assign_clusters(data, centroids)
    
    # Build cluster lists
    clusters = [[] for _ in range(k)]
    for idx, cluster_id in enumerate(assignments):
        clusters[cluster_id].append({
            'point': data[idx],
            'index': idx
        })
    
    # Compute inertia (sum of squared distances to centroid)
    inertia = 0
    for idx, point in enumerate(data):
        cluster_id = assignments[idx]
        inertia += euclidean_distance(point, centroids[cluster_id]) ** 2
    
    # Print results summary
    print(f"\nResults:")
    print(f"  Converged: {converged}")
    print(f"  Iterations: {iterations}")
    print(f"  Inertia (SSE): {inertia:.4f}")
    print(f"\nCluster sizes:")
    for i, cluster in enumerate(clusters):
        print(f"  Cluster {i}: {len(cluster)} points")
    print(f"{'='*60}\n")
    
    return {
        'centroids': centroids,
        'assignments': assignments,
        'clusters': clusters,
        'iterations': iterations,
        'converged': converged,
        'inertia': inertia,
        'k': k
    }


def find_optimal_k(data, k_range=range(2, 11), method='elbow'):
    """Finds the optimal number of clusters using various methods.
    
    Args:
        data (list): List of data points.
        k_range (range): Range of k values to test.
        method (str): Method to use:
            - 'elbow': Elbow method using inertia
            - 'silhouette': Silhouette score (not implemented yet)
    
    Returns:
        dict: Dictionary containing analysis results.
    """
    print(f"\n{'='*60}")
    print(f"Finding Optimal k (method: {method})")
    print(f"Testing k values: {list(k_range)}")
    print(f"{'='*60}")
    
    results = []
    
    for k in k_range:
        result = kmeans(data, k, random_seed=42)
        results.append({
            'k': k,
            'inertia': result['inertia'],
            'converged': result['converged'],
            'iterations': result['iterations']
        })
        print(f"  k={k}: inertia={result['inertia']:.2f}")
    
    # For elbow method, suggest k where rate of decrease slows
    if method == 'elbow' and len(results) >= 3:
        # Compute rate of change
        deltas = []
        for i in range(1, len(results)):
            delta = results[i-1]['inertia'] - results[i]['inertia']
            deltas.append(delta)
        
        # Find "elbow" - where delta decreases significantly
        # Simple heuristic: largest decrease in delta
        if len(deltas) >= 2:
            delta_changes = [deltas[i-1] - deltas[i] for i in range(1, len(deltas))]
            best_idx = delta_changes.index(max(delta_changes)) + 1
            suggested_k = results[best_idx]['k']
        else:
            suggested_k = results[1]['k']
        
        print(f"\nSuggested k (elbow method): {suggested_k}")
    
    return {
        'results': results,
        'suggested_k': suggested_k if method == 'elbow' else None
    }


def convert_to_csv(input_file, output_file=None):
    """Converts various file formats to CSV.
    
    Supported formats:
    - Excel (.xlsx, .xls)
    - JSON (.json) - expects array of objects or 2D array
    - TSV (.tsv, .txt with tabs)
    - Text files with custom delimiters
    
    Args:
        input_file (str): Path to the input file.
        output_file (str, optional): Path to save the CSV file. 
                                     If None, generates name based on input file.
    
    Returns:
        str: Path to the generated CSV file.
    
    Raises:
        ValueError: If file format is not supported.
        ImportError: If required library is not installed.
    """
    import os
    import csv
    import sys
    
    # Get file extension
    _, ext = os.path.splitext(input_file)
    ext = ext.lower()
    
    # Generate output filename if not provided
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_converted.csv"
    
    try:
        if ext in ['.xlsx', '.xls']:
            # Excel files
            try:
                import pandas as pd
            except ImportError:
                print("Error: pandas is required for Excel conversion. Install with: pip install pandas openpyxl", 
                      file=sys.stderr)
                raise
            
            df = pd.read_excel(input_file, header=None)
            df.to_csv(output_file, index=False, header=False)
            
        elif ext == '.json':
            # JSON files
            import json
            
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                if isinstance(data, list):
                    if len(data) > 0:
                        if isinstance(data[0], dict):
                            # List of dictionaries - extract first 2 values
                            for item in data:
                                values = list(item.values())[:2]
                                writer.writerow(values)
                        elif isinstance(data[0], list):
                            # List of lists
                            for row in data:
                                writer.writerow(row[:2])
                        else:
                            raise ValueError("Unsupported JSON structure")
                            
        elif ext in ['.tsv', '.txt']:
            # TSV or text files with tabs/other delimiters
            with open(input_file, 'r', encoding='utf-8') as infile:
                # Try to detect delimiter
                sample = infile.read(1024)
                infile.seek(0)
                
                delimiter = '\t' if '\t' in sample else None
                if delimiter is None:
                    # Try to detect other delimiters
                    for sep in [';', '|', ' ']:
                        if sep in sample:
                            delimiter = sep
                            break
                    if delimiter is None:
                        delimiter = ','
                
                reader = csv.reader(infile, delimiter=delimiter)
                
                with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
                    writer = csv.writer(outfile)
                    for row in reader:
                        if len(row) >= 2:
                            writer.writerow(row[:2])
                            
        else:
            raise ValueError(f"Unsupported file format: {ext}. Supported: .xlsx, .xls, .json, .tsv, .txt")
        
        print(f"Successfully converted '{input_file}' to '{output_file}'")
        return output_file
        
    except Exception as e:
        print(f"Error converting file: {e}", file=sys.stderr)
        raise
    
    
def parse_args():
    """Parses command-line arguments for the CSV file path.

    Returns:
        str: The validated CSV file path.
    """
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(
        description='Clustering & Anomaly Detection on 2D CSV data',
        epilog='Example: python main.py data/data202526a.csv'
    )

    parser.add_argument(
        'filename',
        type=str,
        help='Path to the CSV file containing 2D data points'
    )

    args = parser.parse_args()
    filename = args.filename

    # Validate file exists
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Validate file is readable
    if not os.access(filename, os.R_OK):
        print(f"Error: File '{filename}' is not readable.", file=sys.stderr)
        sys.exit(1)

    # Check if file needs conversion to CSV
    if not filename.lower().endswith('.csv'):
        # Attempt to convert the file to CSV
        supported_formats = ['.xlsx', '.xls', '.json', '.tsv', '.txt']
        _, ext = os.path.splitext(filename)
        
        if ext.lower() in supported_formats:
            print(f"Converting '{filename}' to CSV format...", file=sys.stderr)
            try:
                filename = convert_to_csv(filename)
                print(f"Conversion successful. Using: {filename}", file=sys.stderr)
            except Exception as e:
                print(f"Error: Failed to convert file: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            print(f"Warning: File '{filename}' does not have a .csv extension and is not a supported format for auto-conversion.", file=sys.stderr)
            # Try to proceed anyway - might be a CSV without .csv extension

    return filename
