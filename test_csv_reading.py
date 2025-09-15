#!/usr/bin/env python3
"""
Test CSV reading to ensure Windows paths are preserved correctly
"""

import pandas as pd
import csv
from pathlib import Path

def test_csv_reading_methods(csv_file: str):
    """Test different CSV reading methods"""
    
    print(f"🔍 Testing CSV reading methods for: {csv_file}")
    print("=" * 60)
    
    # Method 1: Standard pandas
    print("\n1️⃣ Standard pandas read_csv:")
    try:
        df1 = pd.read_csv(csv_file)
        print(f"   ✅ Success: {len(df1)} rows")
        if len(df1) > 0:
            print(f"   First path: {df1.iloc[0]['image_path'][:80]}...")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Method 2: Pandas with python engine
    print("\n2️⃣ Pandas with python engine:")
    try:
        df2 = pd.read_csv(csv_file, engine='python', encoding='utf-8')
        print(f"   ✅ Success: {len(df2)} rows")
        if len(df2) > 0:
            print(f"   First path: {df2.iloc[0]['image_path'][:80]}...")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Method 3: Manual CSV parsing
    print("\n3️⃣ Manual CSV parsing:")
    try:
        rows = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, quoting=csv.QUOTE_NONE)
            for line_num, row in enumerate(reader):
                if line_num == 0:
                    continue  # Skip header
                if len(row) >= 2:
                    image_path = row[0].strip()
                    description = ','.join(row[1:]).strip()
                    rows.append({'image_path': image_path, 'description': description})
                    if len(rows) <= 3:  # Show first 3
                        print(f"   Row {len(rows)} path: {image_path[:80]}...")
        
        df3 = pd.DataFrame(rows)
        print(f"   ✅ Success: {len(df3)} rows")
    except Exception as e:
        print(f"   ❌ Failed: {e}")

def create_test_csv():
    """Create a test CSV with Windows paths"""
    
    test_data = [
        {
            'image_path': 'C:\\Users\\danba\\Documents\\VPR_datasets\\SF_XL\\queries_night\\@0543030.55@4180984.98@10@S@37.775128@-122.51158@31760538811@@@@@@20161226@@.jpg',
            'description': 'a red car parked on the street'
        },
        {
            'image_path': 'C:\\Users\\danba\\Documents\\VPR_datasets\\SF_XL\\queries_night\\@0543102.49@4180346.93@10@S@37.76941@-122.510462@28026809684@@@@@@20160730@@.jpg',
            'description': 'a beautiful sunset over the ocean'
        }
    ]
    
    test_csv = 'test_windows_paths.csv'
    df = pd.DataFrame(test_data)
    df.to_csv(test_csv, index=False)
    
    print(f"📝 Created test CSV: {test_csv}")
    print(f"📄 Content preview:")
    print(df.to_string())
    
    return test_csv

if __name__ == "__main__":
    # Create test CSV
    test_file = create_test_csv()
    
    # Test reading methods
    test_csv_reading_methods(test_file)
    
    # Clean up
    Path(test_file).unlink()
    print(f"\n🧹 Cleaned up test file: {test_file}")


