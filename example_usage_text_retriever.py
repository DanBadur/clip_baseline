#!/usr/bin/env python3
"""
Example usage of the Text-to-Image Retriever

This script demonstrates how to use the TextToImageRetriever class
programmatically for text-to-image similarity search.
"""

import pandas as pd
from pathlib import Path
from text_to_image_retriever import TextToImageRetriever
from clip_baseline.config import CLIPConfig

def example_basic_usage():
    """Basic example of text-to-image retrieval"""
    
    print("🔍 Example 1: Basic Text-to-Image Retrieval")
    print("-" * 50)
    
    # Initialize retriever with default config
    retriever = TextToImageRetriever()
    
    # Load image database (assuming you have images in a directory)
    image_database_path = "images"  # Change this to your image directory
    if Path(image_database_path).exists():
        retriever.load_database(image_database_path)
        
        # Search for images using text descriptions
        queries = [
            "a red car",
            "a beautiful landscape",
            "a person smiling",
            "a building with windows"
        ]
        
        print(f"📚 Database loaded with {len(retriever.retriever.image_features)} images")
        
        for query in queries:
            print(f"\n🔍 Searching for: '{query}'")
            results = retriever.retriever.search(query, top_k=3)
            
            for result in results:
                print(f"  {result['rank']}. {result['filename']} (score: {result['similarity_score']:.3f})")
    else:
        print(f"⚠️  Image directory '{image_database_path}' not found. Skipping this example.")

def example_with_custom_config():
    """Example with custom configuration"""
    
    print("\n🔍 Example 2: Custom Configuration")
    print("-" * 50)
    
    # Create custom configuration
    config = CLIPConfig()
    config.model_name = "ViT-B/32"
    config.similarity_metric = "cosine"
    config.default_top_k = 5
    config.verbose = True
    
    # Initialize retriever with custom config
    retriever = TextToImageRetriever(config=config)
    
    # Show configuration details
    print(f"📋 Configuration:")
    print(f"   Model: {config.model_name}")
    print(f"   Similarity metric: {config.similarity_metric}")
    print(f"   Default top-k: {config.default_top_k}")

def example_csv_processing():
    """Example of processing CSV queries"""
    
    print("\n🔍 Example 3: CSV Processing")
    print("-" * 50)
    
    # Create sample queries CSV
    sample_queries = pd.DataFrame({
        'image_path': ['query1.jpg', 'query2.jpg', 'query3.jpg'],
        'description': [
            'a red vehicle on the road',
            'blue water or ocean scene',
            'green trees and nature'
        ]
    })
    
    sample_csv = "sample_queries_example.csv"
    sample_queries.to_csv(sample_csv, index=False)
    print(f"📝 Created sample queries CSV: {sample_csv}")
    
    # Initialize retriever
    retriever = TextToImageRetriever()
    
    # Check if we have a database to work with
    if Path("images").exists():
        retriever.load_database("images")
        
        # Process the CSV
        output_csv = "example_results.csv"
        print(f"🔍 Processing queries from CSV...")
        
        try:
            stats = retriever.process_csv_queries(sample_csv, output_csv, top_k=3)
            print(f"✅ Processing completed!")
            print(f"📊 Statistics: {stats}")
            
            # Show results
            results_df = pd.read_csv(output_csv)
            print(f"\n📋 Results preview:")
            print(results_df.head(10))
            
        except Exception as e:
            print(f"⚠️  Error processing CSV: {e}")
    else:
        print(f"⚠️  Image directory 'images' not found. Skipping CSV processing example.")

def example_database_management():
    """Example of database management"""
    
    print("\n🔍 Example 4: Database Management")
    print("-" * 50)
    
    retriever = TextToImageRetriever()
    
    # Example of building and saving database
    image_dir = "images"
    if Path(image_dir).exists():
        print(f"📚 Building database from: {image_dir}")
        retriever.load_database(image_dir)
        
        # Save database for future use
        db_save_path = "saved_database.npz"
        retriever.save_database(db_save_path)
        print(f"💾 Database saved to: {db_save_path}")
        
        # Get database information
        db_info = retriever.get_database_info()
        print(f"📊 Database info: {db_info}")
        
        # Example of loading saved database
        print(f"\n📚 Loading saved database...")
        new_retriever = TextToImageRetriever()
        new_retriever.load_database(db_save_path)
        
        # Verify it's the same
        new_db_info = new_retriever.get_database_info()
        print(f"📊 Loaded database info: {new_db_info}")
        
    else:
        print(f"⚠️  Image directory '{image_dir}' not found. Skipping database management example.")

def main():
    """Run all examples"""
    
    print("🚀 Text-to-Image Retriever Examples")
    print("=" * 60)
    
    try:
        # Run examples
        example_basic_usage()
        example_with_custom_config()
        example_csv_processing()
        example_database_management()
        
        print(f"\n✅ All examples completed!")
        print(f"📁 Check the generated files:")
        print(f"   - sample_queries_example.csv")
        print(f"   - example_results.csv (if images directory exists)")
        print(f"   - saved_database.npz (if images directory exists)")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print(f"Make sure you have installed all required dependencies:")
        print(f"pip install torch torchvision transformers Pillow numpy scikit-learn tqdm openai-clip ftfy regex pandas")

if __name__ == "__main__":
    main()

