#!/usr/bin/env python3
"""
CSV Diagnostic Tool

This script helps diagnose and fix CSV formatting issues that might prevent
the text-to-image retriever from reading your CSV file properly.
"""

import sys
import csv
import pandas as pd
from pathlib import Path

def diagnose_csv(csv_file: str):
    """Diagnose CSV file for common formatting issues"""
    
    print(f"🔍 Diagnosing CSV file: {csv_file}")
    print("=" * 60)
    
    if not Path(csv_file).exists():
        print(f"❌ File not found: {csv_file}")
        return
    
    # Check file size and basic info
    file_size = Path(csv_file).stat().st_size
    print(f"📊 File size: {file_size:,} bytes")
    
    # Try to read with pandas first
    print(f"\n📖 Attempting pandas read...")
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ Pandas read successful: {len(df)} rows, {len(df.columns)} columns")
        print(f"📋 Columns: {list(df.columns)}")
        print(f"📄 First few rows:")
        print(df.head(3).to_string())
        return
    except Exception as e:
        print(f"❌ Pandas read failed: {e}")
    
    # Manual analysis
    print(f"\n🔧 Manual CSV analysis...")
    
    with open(csv_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    print(f"📊 Total lines: {len(lines)}")
    
    # Check first few lines
    print(f"\n📄 First 5 lines:")
    for i, line in enumerate(lines[:5], 1):
        print(f"{i:2d}: {repr(line[:100])}")
    
    # Check for problematic lines
    print(f"\n🔍 Checking for problematic lines...")
    
    problematic_lines = []
    delimiter_counts = {}
    
    for line_num, line in enumerate(lines, 1):
        # Count different delimiters
        comma_count = line.count(',')
        semicolon_count = line.count(';')
        tab_count = line.count('\t')
        
        delimiter_counts[line_num] = {
            'commas': comma_count,
            'semicolons': semicolon_count,
            'tabs': tab_count
        }
        
        # Check for inconsistent field counts (assuming comma-separated)
        fields = line.split(',')
        if len(fields) > 3:  # More than expected
            problematic_lines.append({
                'line': line_num,
                'field_count': len(fields),
                'content': line.strip()[:100]
            })
    
    # Report delimiter analysis
    print(f"\n📊 Delimiter analysis (first 10 lines):")
    for line_num in range(1, min(11, len(lines) + 1)):
        counts = delimiter_counts[line_num]
        print(f"  Line {line_num:3d}: commas={counts['commas']:2d}, semicolons={counts['semicolons']:2d}, tabs={counts['tabs']:2d}")
    
    # Report problematic lines
    if problematic_lines:
        print(f"\n⚠️  Found {len(problematic_lines)} problematic lines:")
        for prob in problematic_lines[:10]:  # Show first 10
            print(f"  Line {prob['line']:3d}: {prob['field_count']} fields - {prob['content']}")
        if len(problematic_lines) > 10:
            print(f"  ... and {len(problematic_lines) - 10} more")
    else:
        print(f"\n✅ No obviously problematic lines found")
    
    # Try different parsing approaches
    print(f"\n🔧 Testing different parsing approaches...")
    
    # Try with different delimiters
    for delimiter in [',', ';', '\t']:
        try:
            test_df = pd.read_csv(csv_file, delimiter=delimiter)
            print(f"✅ Success with '{delimiter}' delimiter: {len(test_df)} rows, {len(test_df.columns)} columns")
            if len(test_df.columns) == 2:
                print(f"   Columns: {list(test_df.columns)}")
                break
        except Exception as e:
            print(f"❌ Failed with '{delimiter}' delimiter: {e}")

def fix_csv(csv_file: str, output_file: str = None):
    """Attempt to fix CSV formatting issues"""
    
    if output_file is None:
        output_file = csv_file.replace('.csv', '_fixed.csv')
    
    print(f"🔧 Attempting to fix CSV: {csv_file} -> {output_file}")
    
    fixed_rows = []
    skipped_rows = 0
    
    with open(csv_file, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        
        for line_num, row in enumerate(reader, 1):
            if line_num == 1:
                # Header
                fixed_rows.append(['image_path', 'description'])
                continue
            
            if len(row) < 2:
                print(f"⚠️  Skipping line {line_num}: insufficient columns")
                skipped_rows += 1
                continue
            
            if len(row) > 2:
                # Combine extra columns into description
                image_path = row[0].strip()
                description = ','.join(row[1:]).strip()
            else:
                image_path = row[0].strip()
                description = row[1].strip()
            
            # Clean up quotes
            image_path = image_path.strip('"\'')
            description = description.strip('"\'')
            
            fixed_rows.append([image_path, description])
    
    # Write fixed CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(fixed_rows)
    
    print(f"✅ Fixed CSV saved: {output_file}")
    print(f"📊 Processed {len(fixed_rows)} rows, skipped {skipped_rows} rows")
    
    return output_file

def main():
    if len(sys.argv) < 2:
        print("Usage: python csv_diagnostic.py <csv_file> [--fix]")
        print("Examples:")
        print("  python csv_diagnostic.py descriptions.csv")
        print("  python csv_diagnostic.py descriptions.csv --fix")
        return
    
    csv_file = sys.argv[1]
    fix_mode = '--fix' in sys.argv
    
    if fix_mode:
        fixed_file = fix_csv(csv_file)
        print(f"\n🔍 Now diagnosing the fixed file...")
        diagnose_csv(fixed_file)
    else:
        diagnose_csv(csv_file)

if __name__ == "__main__":
    main()



