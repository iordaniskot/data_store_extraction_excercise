import utils

@utils.execution_timer
def main():
    """
    Main entry point for the clustering and anomaly detection program.
    """
    # Parse and validate command-line arguments
    csv_filepath = utils.parse_args()
    
    # Read data from CSV file
    raw_data = utils.csv_reader(csv_filepath)
    print(f"Loaded {len(raw_data)} rows from: {csv_filepath}")
    
    # Clean and validate data
    # Options: handle_missing='remove'|'mean'|'median'|'zero', normalize=True|False
    data_result = utils.clean_and_validate_data(
        raw_data, 
        handle_missing='remove',
        normalize=False  # Set to True if you want normalization
    )
    
    cleaned_data = data_result['cleaned_data']
    original_data = data_result['original_data']
    normalization_params = data_result['normalization_params']
    
    print(f"Ready for processing: {len(cleaned_data)} valid data points")
    
    # ==========================================================================
    # K-MEANS CLUSTERING
    # ==========================================================================
    # Configuration:
    #   k: number of clusters (5 as specified in exercise instructions)
    #   init_method: 'kmeans++' (recommended), 'random', or 'uniform'
    #   max_iterations: maximum iterations before stopping
    #   tolerance: convergence threshold for centroid movement
    
    K = 5  # Number of clusters - specified in exercise instructions
    
    kmeans_result = utils.kmeans(
        cleaned_data,
        k=K,
        max_iterations=100,
        tolerance=1e-6,
        init_method='kmeans++',
        random_seed=42  # For reproducibility
    )
    
    centroids = kmeans_result['centroids']
    assignments = kmeans_result['assignments']
    clusters = kmeans_result['clusters']
    
    # ==========================================================================
    # OUTLIER DETECTION
    # ==========================================================================
    # Detection rule: Distance-based outlier detection
    # A point is considered an outlier if its distance to its assigned centroid
    # exceeds mean_distance + threshold * std_distance
    #
    # Justification:
    # - Points far from any cluster center don't fit the normal data distribution
    # - Using mean + k*std provides an adaptive threshold based on data spread
    # - threshold=2.0 corresponds to ~95% confidence interval (normal distribution)
    #
    # Alternative methods available:
    # - 'iqr': IQR-based detection (robust to extreme outliers)
    # - 'percentile': Top n% furthest points
    # - 'small_cluster': Points in very small clusters
    
    outlier_result = utils.detect_outliers(
        cleaned_data,
        kmeans_result,
        original_data=original_data,  # For printing original coordinates
        method='distance',            # Detection method
        threshold=2.0                 # Number of standard deviations
    )
    
    # Print outliers with original coordinates
    utils.print_outliers(outlier_result['outliers'], show_all=True)
    
    # Get summary for further processing if needed
    outlier_coords = utils.get_outlier_summary(outlier_result['outliers'], original_data)
    
    print(f"\nProgram completed successfully.")
    print(f"  Total data points: {len(cleaned_data)}")
    print(f"  Clusters: {K}")
    print(f"  Outliers detected: {outlier_result['stats']['outlier_count']}")

    utils.plot_clusters(cleaned_data, kmeans_result, outliers=outlier_result['outliers'])


if __name__ == "__main__":
    main()
