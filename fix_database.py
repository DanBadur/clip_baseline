#!/usr/bin/env python3
"""
Script to check and fix duplicate issues in CLIP databases
"""

import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Fix duplicate issues in CLIP database")
    parser.add_argument("--database_path", type=str, required=True, help="Path to the .npz database file")
    parser.add_argument("--check_only", action="store_true", help="Only check, don't fix")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.database_path):
        print(f"❌ Database file not found: {args.database_path}")
        return
    
    print(f"🔍 Checking database: {args.database_path}")
    
    # Import the CLIP retriever
    try:
        sys.path.append('clip_baseline')
        from clip_retriever import CLIPRetriever
    except ImportError as e:
        print(f"❌ Error importing CLIP retriever: {e}")
        print("Make sure you're in the correct directory and clip_baseline is available")
        return
    
    # Initialize retriever and load database
    print("📥 Loading database...")
    retriever = CLIPRetriever()
    
    try:
        retriever.load_database(args.database_path)
    except Exception as e:
        print(f"❌ Error loading database: {e}")
        return
    
    # Check integrity
    print("\n🔍 Checking database integrity...")
    integrity_ok = retriever.check_database_integrity()
    
    if not integrity_ok:
        print("\n⚠️  Database has issues!")
        
        if not args.check_only:
            print("\n🧹 Fixing duplicates...")
            retriever.remove_duplicate_features()
            
            # Save the cleaned database
            backup_path = args.database_path.replace('.npz', '_backup.npz')
            print(f"💾 Creating backup: {backup_path}")
            os.rename(args.database_path, backup_path)
            
            cleaned_path = args.database_path
            print(f"💾 Saving cleaned database: {cleaned_path}")
            retriever.save_database(cleaned_path)
            
            print("\n✅ Database cleaned and saved!")
            print(f"   Original: {backup_path}")
            print(f"   Cleaned: {cleaned_path}")
        else:
            print("\n💡 Use --check_only=False to automatically fix the issues")
    else:
        print("\n✅ Database is clean!")
    
    print(f"\n📊 Final database stats:")
    print(f"   Images: {len(retriever.image_features)}")
    print(f"   Paths: {len(retriever.image_paths)}")
    print(f"   Metadata: {len(retriever.image_metadata)}")

if __name__ == "__main__":
    main()

