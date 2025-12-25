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
    
    # TODO: Implement k-means clustering
    # TODO: Implement anomaly detection
    # TODO: Print outliers in original coordinates
    # Note: Use original_data or denormalize_point() to get original coordinates


if __name__ == "__main__":
    main()
