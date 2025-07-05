#!/usr/bin/env python3
"""
Convert timing data from JSON to CSV format.
Supports single files or glob patterns for multiple files.
"""

import json
import csv
import argparse
from pathlib import Path
import glob
import sys


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Convert timing JSON data to CSV format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Single file
  python timing_to_csv.py input.json -o output.csv
  
  # Multiple files with glob
  python timing_to_csv.py "*.json" -o combined.csv
  
  # Multiple files to separate CSVs
  python timing_to_csv.py "results/*.json" --separate
        '''
    )
    
    parser.add_argument(
        'input',
        help='Input JSON file or glob pattern (e.g., "*.json", "data/*.json")'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='timing_data.csv',
        help='Output CSV file (default: timing_data.csv)'
    )
    
    parser.add_argument(
        '--separate',
        action='store_true',
        help='Create separate CSV files for each input file instead of combining'
    )
    
    parser.add_argument(
        '--add-source',
        action='store_true',
        help='Add source filename column to track which JSON file each row came from'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    return parser.parse_args()


def process_json_file(filepath, add_source=False):
    """Process a single JSON file and return rows for CSV."""
    rows = []
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Process each level
    for level_name in ['level1', 'level2', 'level3']:
        if level_name in data:
            level_data = data[level_name]
            
            # Process each entry in the level
            for entry_name, entry_data in level_data.items():
                row = {
                    'level': level_name,
                    'name': entry_name,
                    'mean': entry_data['mean'],
                    'std': entry_data['std'],
                    'min': entry_data['min'],
                    'max': entry_data['max'],
                    'num_trials': entry_data['num_trials'],
                    'hardware': entry_data['hardware'],
                    'device': entry_data['device']
                }
                
                if add_source:
                    row['source_file'] = filepath.name
                
                rows.append(row)
    
    return rows


def write_csv(rows, output_file, add_source=False):
    """Write rows to CSV file."""
    if not rows:
        print(f"Warning: No data to write to {output_file}")
        return
    
    # Determine fieldnames
    fieldnames = ['level', 'name', 'mean', 'std', 'min', 'max', 'num_trials', 'hardware', 'device']
    if add_source:
        fieldnames.append('source_file')
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_arguments()
    
    # Find all matching files
    if '*' in args.input or '?' in args.input:
        # It's a glob pattern
        files = [Path(f) for f in glob.glob(args.input)]
        if not files:
            print(f"Error: No files matching pattern '{args.input}'")
            sys.exit(1)
    else:
        # Single file
        files = [Path(args.input)]
        if not files[0].exists():
            print(f"Error: File '{args.input}' not found")
            sys.exit(1)
    
    if args.verbose:
        print(f"Found {len(files)} file(s) to process")
    
    # Process files
    if args.separate:
        # Create separate CSV for each JSON file
        for filepath in files:
            try:
                if args.verbose:
                    print(f"Processing {filepath}...")
                
                rows = process_json_file(filepath, add_source=False)
                
                # Generate output filename
                output_file = filepath.with_suffix('.csv')
                write_csv(rows, output_file, add_source=False)
                
                print(f"Created {output_file} with {len(rows)} entries")
                
            except json.JSONDecodeError as e:
                print(f"Error: Invalid JSON in {filepath} - {e}")
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
    
    else:
        # Combine all files into one CSV
        all_rows = []
        
        for filepath in files:
            try:
                if args.verbose:
                    print(f"Processing {filepath}...")
                
                rows = process_json_file(filepath, add_source=args.add_source)
                all_rows.extend(rows)
                
                if args.verbose:
                    print(f"  Added {len(rows)} entries from {filepath}")
                
            except json.JSONDecodeError as e:
                print(f"Error: Invalid JSON in {filepath} - {e}")
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
        
        # Write combined CSV
        write_csv(all_rows, args.output, add_source=args.add_source)
        print(f"Created {args.output} with {len(all_rows)} total entries from {len(files)} file(s)")


if __name__ == "__main__":
    main()