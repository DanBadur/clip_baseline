#!/usr/bin/env python3
"""
Text-to-Image Retriever for VPR (Visual Place Recognition) Analysis

This script creates CSV output compatible with VPR visualization tools.
It extracts UTM coordinates from image filenames and creates reference data
for VPR analysis and visualization.

Input CSV format: image_path,description
Output CSV format: query_image_path,query_description,utm_east,utm_north,reference_1_path,reference_1_score,reference_2_path,reference_2_score,...
"""

import os
import csv
import argparse
import torch
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
import time
import re

# Import the existing CLIP components
try:
    from clip_baseline.clip_retriever import CLIPRetriever
    from clip_baseline.config import CLIPConfig, CONFIG_PRESETS
except ImportError:
    # Fallback for when running from the same directory
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'clip_baseline'))
    from clip_retriever import CLIPRetriever
    from config import CLIPConfig, CONFIG_PRESETS


class TextToImageRetrieverVPR:
    """CLIP-based text-to-image retrieval system for VPR analysis"""
    
    def __init__(self, config: CLIPConfig = None, model_name: str = None, device: str = None):
        """
        Initialize the text-to-image retriever for VPR
        
        Args:
            config: Configuration object (if None, uses VPR config)
            model_name: CLIP model variant to use (overrides config if provided)
            device: Device to run inference on (overrides config if provided)
        """
        # Use VPR config by default
        if config is None:
            config = CONFIG_PRESETS.get('vpr', CONFIG_PRESETS['default'])()
        
        self.retriever = CLIPRetriever(config=config, model_name=model_name, device=device)
        self.config = self.retriever.config
        
    def load_database(self, database_path: str) -> None:
        """
        Load image database from directory or saved database file
        
        Args:
            database_path: Path to image directory or saved .npz database file
        """
        if database_path.endswith('.npz'):
            # Load from saved database
            self.retriever.load_database(database_path)
        else:
            # Build database from image directory
            self.retriever.build_image_database(database_path)
    
    def extract_utm_coordinates(self, image_path: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Extract UTM coordinates from image filename or path
        
        This function looks for UTM coordinates in various formats:
        - @east@north@ format (e.g., @0543256.96@4178906.70@)
        - filename with numbers that could be coordinates
        - GPS coordinates that can be converted to UTM
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (utm_east, utm_north) or (None, None) if not found
        """
        filename = Path(image_path).name
        
        # Try to extract UTM coordinates from filename patterns
        # Pattern 1: @east@north@ format
        utm_pattern = r'@([+-]?\d+\.?\d*)@([+-]?\d+\.?\d*)@'
        match = re.search(utm_pattern, filename)
        if match:
            try:
                east = float(match.group(1))
                north = float(match.group(2))
                return east, north
            except ValueError:
                pass
        
        # Pattern 2: Look for two large numbers that could be UTM coordinates
        # UTM coordinates are typically 6-7 digits
        number_pattern = r'(\d{6,7}\.?\d*)'
        numbers = re.findall(number_pattern, filename)
        if len(numbers) >= 2:
            try:
                east = float(numbers[0])
                north = float(numbers[1])
                # Basic validation: UTM coordinates should be reasonable
                if 100000 <= east <= 1000000 and 0 <= north <= 10000000:
                    return east, north
            except ValueError:
                pass
        
        # Pattern 3: Look for GPS coordinates (lat, lon) and convert to UTM
        # This is a simplified conversion - for production use a proper library
        gps_pattern = r'([+-]?\d+\.\d+),([+-]?\d+\.\d+)'
        match = re.search(gps_pattern, filename)
        if match:
            try:
                lat = float(match.group(1))
                lon = float(match.group(2))
                # Simple approximation for UTM Zone 10N (adjust as needed)
                utm_east = (lon + 123) * 111320 * np.cos(np.radians(lat))
                utm_north = (lat - 49) * 111320
                return utm_east, utm_north
            except ValueError:
                pass
        
        return None, None
    
    def process_csv_queries_vpr(self, input_csv: str, output_csv: str, top_k: int = 5) -> Dict:
        """
        Process CSV file with queries and save VPR-compatible results to output CSV
        
        Args:
            input_csv: Path to input CSV with columns: image_path, description
            output_csv: Path to output CSV file for VPR visualization
            top_k: Number of top similar images to retrieve for each query
            
        Returns:
            Dictionary with processing statistics
        """
        print(f"📖 Reading input CSV: {input_csv}")
        
        # Read input CSV
        try:
            df = pd.read_csv(input_csv)
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")
        
        # Validate CSV format
        required_columns = ['image_path', 'description']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"CSV missing required columns: {missing_columns}")
        
        print(f"✅ Found {len(df)} queries in CSV")
        
        # Prepare output data
        output_data = []
        processing_stats = {
            'total_queries': len(df),
            'successful_queries': 0,
            'failed_queries': 0,
            'total_retrievals': 0,
            'processing_time': 0,
            'utm_extraction_success': 0
        }
        
        start_time = time.time()
        
        print(f"🔍 Processing queries with top_k={top_k}...")
        
        # Process each query
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing queries"):
            query_image_path = row['image_path']
            query_description = row['description']
            
            try:
                # Extract UTM coordinates from query image path
                utm_east, utm_north = self.extract_utm_coordinates(query_image_path)
                if utm_east is not None and utm_north is not None:
                    processing_stats['utm_extraction_success'] += 1
                
                # Search for similar images using text description
                results = self.retriever.search(query_description, top_k=top_k)
                
                # Prepare row data for VPR CSV
                row_data = {
                    'query_image_path': query_image_path,
                    'query_description': query_description,
                    'utm_east': utm_east if utm_east is not None else 0.0,
                    'utm_north': utm_north if utm_north is not None else 0.0
                }
                
                # Add reference image data (top-k results)
                for i, result in enumerate(results):
                    ref_idx = i + 1
                    row_data[f'reference_{ref_idx}_path'] = result['image_path']
                    row_data[f'reference_{ref_idx}_filename'] = result['filename']
                    row_data[f'reference_{ref_idx}_score'] = result['similarity_score']
                    row_data[f'reference_{ref_idx}_size'] = result['metadata']['size']
                
                # Fill remaining reference columns with empty values if we have fewer than top_k results
                for i in range(len(results), top_k):
                    ref_idx = i + 1
                    row_data[f'reference_{ref_idx}_path'] = ""
                    row_data[f'reference_{ref_idx}_filename'] = ""
                    row_data[f'reference_{ref_idx}_score'] = 0.0
                    row_data[f'reference_{ref_idx}_size'] = ""
                
                output_data.append(row_data)
                processing_stats['successful_queries'] += 1
                processing_stats['total_retrievals'] += len(results)
                
            except Exception as e:
                print(f"⚠️  Error processing query {idx+1}: {e}")
                processing_stats['failed_queries'] += 1
                continue
        
        processing_stats['processing_time'] = time.time() - start_time
        
        # Save results to CSV
        print(f"💾 Saving VPR-compatible results to: {output_csv}")
        output_df = pd.DataFrame(output_data)
        output_df.to_csv(output_csv, index=False)
        
        print(f"✅ VPR results saved successfully!")
        print(f"📊 Processing Statistics:")
        print(f"   Total queries: {processing_stats['total_queries']}")
        print(f"   Successful: {processing_stats['successful_queries']}")
        print(f"   Failed: {processing_stats['failed_queries']}")
        print(f"   Total retrievals: {processing_stats['total_retrievals']}")
        print(f"   UTM extraction success: {processing_stats['utm_extraction_success']}")
        print(f"   Processing time: {processing_stats['processing_time']:.2f}s")
        
        return processing_stats
    
    def save_database(self, save_path: str) -> None:
        """Save the current image database"""
        self.retriever.save_database(save_path)
    
    def get_database_info(self) -> Dict:
        """Get information about the loaded database"""
        if self.retriever.image_features is None:
            return {'status': 'No database loaded'}
        
        return {
            'status': 'Database loaded',
            'total_images': len(self.retriever.image_features),
            'feature_dimension': self.retriever.image_features.shape[1],
            'device': str(self.retriever.image_features.device),
            'model_name': self.config.model_name,
            'similarity_metric': self.config.similarity_metric
        }


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description="CLIP-based Text-to-Image Retrieval for VPR Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic VPR usage
  python text_to_image_retriever_vpr.py --input queries.csv --database ./images --output vpr_results.csv

  # With custom top-k and VPR preset
  python text_to_image_retriever_vpr.py --input queries.csv --database ./images --output vpr_results.csv --top_k 10 --preset vpr

  # Using saved database
  python text_to_image_retriever_vpr.py --input queries.csv --database ./saved_db.npz --output vpr_results.csv
        """
    )
    
    # Required arguments
    parser.add_argument("--input", type=str, required=True,
                       help="Input CSV file with columns: image_path, description")
    parser.add_argument("--database", type=str, required=True,
                       help="Image database directory or saved .npz database file")
    parser.add_argument("--output", type=str, required=True,
                       help="Output CSV file for VPR visualization")
    
    # Optional arguments
    parser.add_argument("--top_k", type=int, default=5,
                       help="Number of top similar images to retrieve (default: 5)")
    parser.add_argument("--config", type=str, default=None,
                       help="Configuration file path")
    parser.add_argument("--preset", type=str, choices=list(CONFIG_PRESETS.keys()),
                       help="Use predefined configuration preset (default: vpr)")
    parser.add_argument("--model", type=str, default=None,
                       help="CLIP model variant (e.g., ViT-B/32, ViT-L/14)")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda, cpu, or auto-detect)")
    parser.add_argument("--save_database", type=str, default=None,
                       help="Path to save database after building (for image directories)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file not found: {args.input}")
        return 1
    
    # Validate database path
    if not os.path.exists(args.database):
        print(f"❌ Error: Database path not found: {args.database}")
        return 1
    
    # Load configuration (use VPR preset by default)
    try:
        if args.preset:
            config = CONFIG_PRESETS[args.preset]()
            if args.verbose:
                print(f"📋 Using preset configuration: {args.preset}")
        elif args.config and os.path.exists(args.config):
            config = CLIPConfig.from_file(args.config)
            if args.verbose:
                print(f"📋 Using configuration file: {args.config}")
        else:
            # Default to VPR configuration
            config = CONFIG_PRESETS['vpr']()
            if args.verbose:
                print("📋 Using VPR configuration (default)")
    except Exception as e:
        print(f"⚠️  Error loading config: {e}, using VPR defaults")
        config = CONFIG_PRESETS['vpr']()
    
    # Override config with command line arguments
    if args.model:
        config.model_name = args.model
    if args.device:
        config.device = args.device
    if args.verbose:
        config.verbose = True
    
    # Initialize retriever
    print("🚀 Initializing CLIP Text-to-Image Retriever for VPR...")
    retriever = TextToImageRetrieverVPR(config=config)
    
    # Load database
    print(f"📚 Loading database from: {args.database}")
    retriever.load_database(args.database)
    
    # Show database info
    db_info = retriever.get_database_info()
    if args.verbose:
        print(f"📊 Database Info: {db_info}")
    
    # Save database if requested (only for image directories)
    if args.save_database and not args.database.endswith('.npz'):
        print(f"💾 Saving database to: {args.save_database}")
        retriever.save_database(args.save_database)
    
    # Process queries
    print(f"\n🔍 Starting text-to-image retrieval for VPR analysis...")
    stats = retriever.process_csv_queries_vpr(args.input, args.output, top_k=args.top_k)
    
    print(f"\n✅ VPR processing completed successfully!")
    print(f"📁 VPR-compatible results saved to: {args.output}")
    print(f"📊 UTM coordinate extraction: {stats['utm_extraction_success']}/{stats['total_queries']} successful")
    
    return 0


if __name__ == "__main__":
    exit(main())



