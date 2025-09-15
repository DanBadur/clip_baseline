#!/usr/bin/env python3
"""
Diagnostic script to check why more images are found than expected
"""

import os
from pathlib import Path
from collections import defaultdict

def diagnose_image_directory(image_dir: str):
    """Diagnose image directory to understand the counting discrepancy"""
    
    print(f"🔍 Diagnosing image directory: {image_dir}")
    print("=" * 60)
    
    # Supported formats
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    image_dir = Path(image_dir)
    if not image_dir.exists():
        print(f"❌ Directory does not exist: {image_dir}")
        return
    
    # Count files by extension
    extension_counts = defaultdict(int)
    all_files = []
    duplicate_files = []
    
    print(f"📁 Scanning directory recursively...")
    
    # Find all image files (both cases)
    for ext in supported_formats:
        # Lowercase
        files_lower = list(image_dir.rglob(f"*{ext}"))
        for file_path in files_lower:
            extension_counts[ext] += 1
            all_files.append(str(file_path))
        
        # Uppercase
        files_upper = list(image_dir.rglob(f"*{ext.upper()}"))
        for file_path in files_upper:
            extension_counts[ext.upper()] += 1
            all_files.append(str(file_path))
    
    # Find duplicates (same file found with different case extensions)
    file_paths_lower = set()
    file_paths_upper = set()
    
    for ext in supported_formats:
        files_lower = list(image_dir.rglob(f"*{ext}"))
        files_upper = list(image_dir.rglob(f"*{ext.upper()}"))
        
        for file_path in files_lower:
            file_paths_lower.add(str(file_path))
        for file_path in files_upper:
            file_paths_upper.add(str(file_path))
    
    duplicates = file_paths_lower.intersection(file_paths_upper)
    
    # Results
    print(f"\n📊 Results:")
    print(f"   Total files found: {len(all_files)}")
    print(f"   Unique files: {len(set(all_files))}")
    print(f"   Duplicates due to case: {len(duplicates)}")
    
    print(f"\n📋 Extension breakdown:")
    for ext, count in sorted(extension_counts.items()):
        print(f"   {ext}: {count} files")
    
    if duplicates:
        print(f"\n⚠️  Duplicate files (found with both lowercase and uppercase extensions):")
        for dup in sorted(list(duplicates))[:10]:  # Show first 10
            print(f"   {dup}")
        if len(duplicates) > 10:
            print(f"   ... and {len(duplicates) - 10} more")
    
    # Show some example files
    print(f"\n📄 Sample files found:")
    for i, file_path in enumerate(sorted(set(all_files))[:10]):
        print(f"   {i+1}. {file_path}")
    if len(set(all_files)) > 10:
        print(f"   ... and {len(set(all_files)) - 10} more")
    
    return {
        'total_found': len(all_files),
        'unique_files': len(set(all_files)),
        'duplicates': len(duplicates),
        'extension_counts': dict(extension_counts)
    }

def main():
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python diagnose_image_count.py <image_directory>")
        print("Example: python diagnose_image_count.py ./images")
        return
    
    image_dir = sys.argv[1]
    diagnose_image_directory(image_dir)

if __name__ == "__main__":
    main()

