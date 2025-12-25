

def csv_reader(file_path):
    """Reads a CSV file and returns its content as a list of dictionaries.

    Args:
        file_path (str): The path to the CSV file.
    Returns:
        list: A list of dictionaries representing the rows in the CSV file.
    """
    import csv

    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        return [row for row in reader]
    
    
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

    # Optional: Validate .csv extension
    if not filename.lower().endswith('.csv'):
        print(f"Warning: File '{filename}' does not have a .csv extension.", file=sys.stderr)
        # Note: Only warning, not failing - could be a valid CSV without .csv extension

    return filename
