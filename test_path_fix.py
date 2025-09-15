#!/usr/bin/env python3
"""
Test the path fixing logic with the user's specific example
"""

import re

def fix_single_path(path: str) -> str:
    """
    Fix a single Windows path by adding missing backslashes
    
    Args:
        path: Original path string
        
    Returns:
        Fixed path string
    """
    
    print(f"Original: {path}")
    
    # If path already has backslashes, return as-is
    if '\\' in path:
        print("Already has backslashes")
        return path
    
    # If it's a Windows-style path (has drive letter and colon)
    if re.match(r'^[A-Za-z]:', path):
        # Fix drive letter followed by letters (C:Users -> C:\Users)
        path = re.sub(r'([A-Za-z]:)([A-Za-z])', r'\1\\\2', path)
        print(f"After drive letter fix: {path}")
        
        # Fix camelCase directory names (UsersDocuments -> Users\Documents)
        path = re.sub(r'([a-z])([A-Z])', r'\1\\\2', path)
        print(f"After camelCase fix: {path}")
        
        # Fix capital letters followed by lowercase (SF_XLqueries -> SF_XL\queries)
        path = re.sub(r'([A-Z])([a-z])', r'\1\\\2', path)
        print(f"After capital-lowercase fix: {path}")
        
        # Fix common directory patterns
        patterns = [
            (r'Documents([A-Z])', r'Documents\\\1'),
            (r'Desktop([A-Z])', r'Desktop\\\1'),
            (r'VPR_datasets([A-Z])', r'VPR_datasets\\\1'),
            (r'queries_([a-z]+)([A-Z])', r'queries_\1\\\2'),
        ]
        
        for pattern, replacement in patterns:
            old_path = path
            path = re.sub(pattern, replacement, path)
            if old_path != path:
                print(f"After pattern '{pattern}': {path}")
    
    print(f"Final: {path}")
    return path

def main():
    # Test with the user's example
    test_path = "C:UsersdanbaDocumentsVPR_datasetsSF_XLqueries_night@0543030.55@4180984.98@10@S@37.775128@-122.51158@31760538811@@@@@@20161226@@.jpg"
    
    print("Testing path fixing logic:")
    print("=" * 60)
    
    fixed_path = fix_single_path(test_path)
    
    print("\nExpected result:")
    expected = "C:\\Users\\danba\\Documents\\VPR_datasets\\SF_XL\\queries_night@0543030.55@4180984.98@10@S@37.775128@-122.51158@31760538811@@@@@@20161226@@.jpg"
    print(expected)
    
    print(f"\nMatch: {fixed_path == expected}")

if __name__ == "__main__":
    main()
