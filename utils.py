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
