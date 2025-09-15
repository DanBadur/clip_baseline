#!/usr/bin/env python3
"""
Test script for the new CSV output structure
"""

import pandas as pd
import numpy as np

def create_sample_output():
    """Create a sample output CSV with the new structure"""
    
    # Sample data with the exact structure requested
    sample_data = [
        {
            'query_image_path': '@0543256.96@4178906.70@image001.jpg',
            'utm_east': 543256.96,
            'utm_north': 4178906.7,
            'reference_1_path': 'database/car_001.jpg',
            'reference_1_distance': 0.1766,
            'reference_1_utm_east': 543260.15,
            'reference_1_utm_north': 4178911.85,
            'reference_2_path': 'database/vehicle_002.jpg',
            'reference_2_distance': 0.2109,
            'reference_2_utm_east': 543262.38,
            'reference_2_utm_north': 4178912.96,
            'reference_3_path': 'database/red_car_003.jpg',
            'reference_3_distance': 0.2346,
            'reference_3_utm_east': 543264.84,
            'reference_3_utm_north': 4178914.18
        },
        {
            'query_image_path': '@0543257.12@4178907.85@image002.jpg',
            'utm_east': 543257.12,
            'utm_north': 4178907.85,
            'reference_1_path': 'database/ocean_001.jpg',
            'reference_1_distance': 0.0844,
            'reference_1_utm_east': 543261.15,
            'reference_1_utm_north': 4178911.85,
            'reference_2_path': 'database/sunset_002.jpg',
            'reference_2_distance': 0.1068,
            'reference_2_utm_east': 543262.38,
            'reference_2_utm_north': 4178912.96,
            'reference_3_path': 'database/water_003.jpg',
            'reference_3_distance': 0.1544,
            'reference_3_utm_east': 543264.84,
            'reference_3_utm_north': 4178914.18
        },
        {
            'query_image_path': 'failed_query.jpg',
            'utm_east': np.nan,
            'utm_north': np.nan,
            'reference_1_path': np.nan,
            'reference_1_distance': np.nan,
            'reference_1_utm_east': np.nan,
            'reference_1_utm_north': np.nan,
            'reference_2_path': np.nan,
            'reference_2_distance': np.nan,
            'reference_2_utm_east': np.nan,
            'reference_2_utm_north': np.nan,
            'reference_3_path': np.nan,
            'reference_3_distance': np.nan,
            'reference_3_utm_east': np.nan,
            'reference_3_utm_north': np.nan
        }
    ]
    
    # Create DataFrame
    df = pd.DataFrame(sample_data)
    
    # Save to CSV
    output_file = 'sample_new_structure_output.csv'
    df.to_csv(output_file, index=False)
    
    print(f"✅ Created sample output CSV: {output_file}")
    print(f"📊 Structure:")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Rows: {len(df)}")
    print(f"   Shape: {df.shape}")
    
    # Show first row
    print(f"\n📄 First row:")
    print(df.iloc[0].to_dict())
    
    # Show failed query row
    print(f"\n❌ Failed query row (with NaN values):")
    print(df.iloc[2].to_dict())
    
    return output_file

if __name__ == "__main__":
    create_sample_output()


