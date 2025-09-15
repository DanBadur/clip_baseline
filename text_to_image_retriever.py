#!/usr/bin/env python3
"""
Text-to-Image Retriever using CLIP

This script takes a CSV file with image paths and descriptions, and for each description,
finds the top-k most similar images from a database using CLIP text-image similarity.

Input CSV format: image_path,description
Output CSV format: query_image_path,query_description,rank,retrieved_image_path,similarity_score
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


class TextToImageRetriever:
    """CLIP-based text-to-image retrieval system"""
    
    def __init__(self, config: CLIPConfig = None, model_name: str = None, device: str = None):
        """
        Initialize the text-to-image retriever
        
        Args:
            config: Configuration object (if None, uses default)
            model_name: CLIP model variant to use (overrides config if provided)
            device: Device to run inference on (overrides config if provided)
        """
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
        import re
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

    def process_csv_queries(self, input_csv: str, output_csv: str, top_k: int = 3) -> Dict:
        """
        Process CSV file with queries and save results to output CSV
        
        Args:
            input_csv: Path to input CSV with columns: image_path, description
            output_csv: Path to output CSV file with structure:
                query_image_path,utm_east,utm_north,reference_1_path,reference_1_distance,reference_1_utm_east,reference_1_utm_north,...
            top_k: Number of top similar images to retrieve for each query
            
        Returns:
            Dictionary with processing statistics
        """
        print(f"📖 Reading input CSV: {input_csv}")
        
        # Read input CSV with proper handling for Windows paths
        try:
            # First, try reading with proper escaping for Windows paths
            df = pd.read_csv(input_csv, 
                           engine='python',  # Use python engine for better path handling
                           encoding='utf-8',
                           on_bad_lines='skip',
                           quotechar='"',
                           skipinitialspace=True)
            print(f"✅ Successfully read CSV with {len(df)} valid rows")
        except Exception as e:
            print(f"⚠️  CSV parsing error: {e}")
            print("🔧 Attempting manual CSV parsing to preserve Windows paths...")
            
            # Manual CSV parsing to preserve backslashes
            df = self._manual_csv_parse(input_csv)
        
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
            'processing_time': 0
        }
        
        start_time = time.time()
        
        print(f"🔍 Processing queries with top_k={top_k}...")
        
        # Process each query
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing queries"):
            query_image_path = row['image_path']
            query_description = row['description']
            
            # Extract UTM coordinates for query image
            query_utm_east, query_utm_north = self.extract_utm_coordinates(query_image_path)
            
            try:
                # Search for similar images using text description
                results = self.retriever.search(query_description, top_k=top_k)
                
                # Prepare row data with the exact structure requested
                row_data = {
                    'query_image_path': query_image_path,
                    'utm_east': query_utm_east if query_utm_east is not None else float('nan'),
                    'utm_north': query_utm_north if query_utm_north is not None else float('nan')
                }
                
                # Add reference data for each result
                for i, result in enumerate(results):
                    ref_idx = i + 1
                    
                    # Extract UTM coordinates for reference image
                    ref_utm_east, ref_utm_north = self.extract_utm_coordinates(result['image_path'])
                    
                    row_data[f'reference_{ref_idx}_path'] = result['image_path']
                    row_data[f'reference_{ref_idx}_distance'] = 1.0 - result['similarity_score']  # Convert similarity to distance
                    row_data[f'reference_{ref_idx}_utm_east'] = ref_utm_east if ref_utm_east is not None else float('nan')
                    row_data[f'reference_{ref_idx}_utm_north'] = ref_utm_north if ref_utm_north is not None else float('nan')
                
                # Fill remaining reference columns with NaN if we have fewer than top_k results
                for i in range(len(results), top_k):
                    ref_idx = i + 1
                    row_data[f'reference_{ref_idx}_path'] = float('nan')
                    row_data[f'reference_{ref_idx}_distance'] = float('nan')
                    row_data[f'reference_{ref_idx}_utm_east'] = float('nan')
                    row_data[f'reference_{ref_idx}_utm_north'] = float('nan')
                
                output_data.append(row_data)
                processing_stats['successful_queries'] += 1
                processing_stats['total_retrievals'] += len(results)
                
            except Exception as e:
                print(f"⚠️  Error processing query {idx+1}: {e}")
                
                # Create row with NaN values for failed queries
                row_data = {
                    'query_image_path': query_image_path,
                    'utm_east': query_utm_east if query_utm_east is not None else float('nan'),
                    'utm_north': query_utm_north if query_utm_north is not None else float('nan')
                }
                
                # Fill all reference columns with NaN for failed queries
                for i in range(top_k):
                    ref_idx = i + 1
                    row_data[f'reference_{ref_idx}_path'] = float('nan')
                    row_data[f'reference_{ref_idx}_distance'] = float('nan')
                    row_data[f'reference_{ref_idx}_utm_east'] = float('nan')
                    row_data[f'reference_{ref_idx}_utm_north'] = float('nan')
                
                output_data.append(row_data)
                processing_stats['failed_queries'] += 1
                continue
        
        processing_stats['processing_time'] = time.time() - start_time
        
        # Save results to CSV
        print(f"💾 Saving results to: {output_csv}")
        output_df = pd.DataFrame(output_data)
        output_df.to_csv(output_csv, index=False)
        
        print(f"✅ Results saved successfully!")
        print(f"📊 Processing Statistics:")
        print(f"   Total queries: {processing_stats['total_queries']}")
        print(f"   Successful: {processing_stats['successful_queries']}")
        print(f"   Failed: {processing_stats['failed_queries']}")
        print(f"   Total retrievals: {processing_stats['total_retrievals']}")
        print(f"   Processing time: {processing_stats['processing_time']:.2f}s")
        
        return processing_stats
    
    def save_database(self, save_path: str) -> None:
        """Save the current image database"""
        self.retriever.save_database(save_path)
    
    def _manual_csv_parse(self, csv_file: str) -> pd.DataFrame:
        """
        Manually parse CSV file to preserve Windows paths correctly
        
        Args:
            csv_file: Path to CSV file
            
        Returns:
            DataFrame with parsed data
        """
        import csv
        
        rows = []
        skipped_rows = 0
        
        print("📖 Manually parsing CSV file to preserve Windows paths...")
        
        with open(csv_file, 'r', encoding='utf-8', errors='ignore') as f:
            # Try to detect the delimiter
            sample = f.read(1024)
            f.seek(0)
            
            sniffer = csv.Sniffer()
            try:
                dialect = sniffer.sniff(sample)
                delimiter = dialect.delimiter
            except:
                delimiter = ','  # Default to comma
            
            # Use a custom reader that preserves backslashes
            reader = csv.reader(f, delimiter=delimiter, quoting=csv.QUOTE_NONE, escapechar='\\')
            
            for line_num, row in enumerate(reader, 1):
                if line_num == 1:
                    # Header row
                    headers = row
                    print(f"📋 Detected headers: {headers}")
                    continue
                
                if len(row) < 2:
                    print(f"⚠️  Skipping line {line_num}: insufficient columns")
                    skipped_rows += 1
                    continue
                
                if len(row) > 2:
                    # If we have more than 2 columns, combine everything after the first column as description
                    image_path = row[0].strip()
                    description = delimiter.join(row[1:]).strip()
                else:
                    image_path = row[0].strip()
                    description = row[1].strip()
                
                # Clean up quotes but preserve everything else including backslashes
                image_path = image_path.strip('"\'')
                description = description.strip('"\'')
                
                # Log first few paths to verify they're correct
                if line_num <= 3:
                    print(f"   Line {line_num} path: {image_path[:80]}...")
                
                rows.append({
                    'image_path': image_path,
                    'description': description
                })
        
        if skipped_rows > 0:
            print(f"⚠️  Skipped {skipped_rows} malformed rows")
        
        df = pd.DataFrame(rows)
        print(f"✅ Manually parsed {len(df)} valid rows")
        
        return df
    
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
        description="CLIP-based Text-to-Image Retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with image directory
  python text_to_image_retriever.py --input queries.csv --database ./images --output results.csv

  # Using saved database
  python text_to_image_retriever.py --input queries.csv --database ./saved_db.npz --output results.csv

  # With custom top-k and configuration
  python text_to_image_retriever.py --input queries.csv --database ./images --output results.csv --top_k 10 --config custom_config.json

  # Using preset configuration
  python text_to_image_retriever.py --input queries.csv --database ./images --output results.csv --preset high_accuracy
        """
    )
    
    # Required arguments
    parser.add_argument("--input", type=str, required=True,
                       help="Input CSV file with columns: image_path, description")
    parser.add_argument("--database", type=str, required=True,
                       help="Image database directory or saved .npz database file")
    parser.add_argument("--output", type=str, required=True,
                       help="Output CSV file for results")
    
    # Optional arguments
    parser.add_argument("--top_k", type=int, default=3,
                       help="Number of top similar images to retrieve (default: 3)")
    parser.add_argument("--config", type=str, default=None,
                       help="Configuration file path")
    parser.add_argument("--preset", type=str, choices=list(CONFIG_PRESETS.keys()),
                       help="Use predefined configuration preset")
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
    
    # Load configuration
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
            config = CLIPConfig()
            if args.verbose:
                print("📋 Using default configuration")
    except Exception as e:
        print(f"⚠️  Error loading config: {e}, using defaults")
        config = CLIPConfig()
    
    # Override config with command line arguments
    if args.model:
        config.model_name = args.model
    if args.device:
        config.device = args.device
    if args.verbose:
        config.verbose = True
    
    # Initialize retriever
    print("🚀 Initializing CLIP Text-to-Image Retriever...")
    retriever = TextToImageRetriever(config=config)
    
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
    print(f"\n🔍 Starting text-to-image retrieval...")
    stats = retriever.process_csv_queries(args.input, args.output, top_k=args.top_k)
    
    print(f"\n✅ Processing completed successfully!")
    print(f"📁 Results saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())
