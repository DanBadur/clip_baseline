#!/usr/bin/env python3
"""
Fix Windows paths in CSV files by adding missing backslashes
"""

import pandas as pd
import re
import sys

def fix_windows_paths(csv_file: str, output_file: str = None):
    """
    Fix Windows paths in CSV file by adding missing backslashes
    
    Args:
        csv_file: Input CSV file path
        output_file: Output CSV file path (if None, overwrites input)
    """
    
    if output_file is None:
        output_file = csv_file.replace('.csv', '_fixed.csv')
    
    print(f"🔧 Fixing Windows paths in: {csv_file}")
    print(f"📁 Output file: {output_file}")
    
    # Read CSV
    df = pd.read_csv(csv_file)
    
    print(f"📊 Original CSV shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    
    # Fix paths in all columns that contain file paths
    path_columns = [col for col in df.columns if 'path' in col.lower()]
    
    print(f"🔍 Found path columns: {path_columns}")
    
    fixed_count = 0
    
    for col in path_columns:
        print(f"🔧 Fixing column: {col}")
        
        for idx, path in enumerate(df[col]):
            if pd.isna(path) or path == '' or path == 'nan':
                continue
                
            original_path = str(path)
            fixed_path = fix_single_path(original_path)
            
            if original_path != fixed_path:
                df.loc[idx, col] = fixed_path
                fixed_count += 1
                if fixed_count <= 5:  # Show first 5 fixes
                    print(f"   {original_path[:50]}... -> {fixed_path[:50]}...")
    
    # Save fixed CSV
    df.to_csv(output_file, index=False)
    
    print(f"✅ Fixed {fixed_count} paths")
    print(f"💾 Saved to: {output_file}")
    
    return output_file

def fix_single_path(path: str) -> str:
    """
    Fix a single Windows path by adding missing backslashes
    
    Args:
        path: Original path string
        
    Returns:
        Fixed path string
    """
    
    # If path already has backslashes, return as-is
    if '\\' in path:
        return path
    
    # If it's a Windows-style path (has drive letter and colon)
    if re.match(r'^[A-Za-z]:', path):
        # Fix drive letter followed by letters (C:Users -> C:\Users)
        path = re.sub(r'([A-Za-z]:)([A-Za-z])', r'\1\\\2', path)
        
        # Fix camelCase directory names (UsersDocuments -> Users\Documents)
        path = re.sub(r'([a-z])([A-Z])', r'\1\\\2', path)
        
        # Fix capital letters followed by lowercase (SF_XLqueries -> SF_XL\queries)
        path = re.sub(r'([A-Z])([a-z])', r'\1\\\2', path)
        
        # Fix common directory patterns
        patterns = [
            (r'Documents([A-Z])', r'Documents\\\1'),
            (r'Desktop([A-Z])', r'Desktop\\\1'),
            (r'VPR_datasets([A-Z])', r'VPR_datasets\\\1'),
            (r'queries_([a-z]+)([A-Z])', r'queries_\1\\\2'),
        ]
        
        for pattern, replacement in patterns:
            path = re.sub(pattern, replacement, path)
    
    return path

def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_paths_in_csv.py <csv_file> [output_file]")
        print("Examples:")
        print("  python fix_paths_in_csv.py results.csv")
        print("  python fix_paths_in_csv.py results.csv results_fixed.csv")
        return
    
    input_csv = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None
    
    fix_windows_paths(input_csv, output_csv)

if __name__ == "__main__":
    main()
