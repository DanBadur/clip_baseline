#!/usr/bin/env python3
"""
Test script for the Text-to-Image Retriever

This script creates a small test dataset and demonstrates the functionality
of the text_to_image_retriever.py script.
"""

import os
import pandas as pd
from PIL import Image
import numpy as np
from pathlib import Path

def create_test_dataset():
    """Create a small test dataset with sample images and queries"""
    
    # Create test directories
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    image_dir = test_dir / "images"
    image_dir.mkdir(exist_ok=True)
    
    # Create sample images (colored rectangles)
    sample_images = [
        ("red_car.jpg", (255, 0, 0), "Red rectangle representing a car"),
        ("blue_ocean.jpg", (0, 0, 255), "Blue rectangle representing ocean"),
        ("green_tree.jpg", (0, 255, 0), "Green rectangle representing tree"),
        ("yellow_sun.jpg", (255, 255, 0), "Yellow rectangle representing sun"),
        ("orange_fruit.jpg", (255, 165, 0), "Orange rectangle representing fruit"),
        ("purple_flower.jpg", (128, 0, 128), "Purple rectangle representing flower"),
        ("brown_bear.jpg", (139, 69, 19), "Brown rectangle representing bear"),
        ("pink_flower.jpg", (255, 192, 203), "Pink rectangle representing flower"),
    ]
    
    print("🎨 Creating test images...")
    for filename, color, description in sample_images:
        # Create a simple colored image
        img = Image.new('RGB', (100, 100), color)
        img.save(image_dir / filename)
        print(f"   Created: {filename} ({description})")
    
    # Create test queries CSV
    queries_data = [
        ("query1.jpg", "a red vehicle"),
        ("query2.jpg", "blue water or ocean"),
        ("query3.jpg", "green nature or plants"),
        ("query4.jpg", "bright yellow object"),
        ("query5.jpg", "orange colored item"),
    ]
    
    queries_df = pd.DataFrame(queries_data, columns=['image_path', 'description'])
    queries_csv = test_dir / "test_queries.csv"
    queries_df.to_csv(queries_csv, index=False)
    
    print(f"📝 Created test queries CSV: {queries_csv}")
    print(f"📁 Test dataset created in: {test_dir}")
    
    return test_dir, queries_csv, image_dir

def run_test():
    """Run a test of the text-to-image retriever"""
    
    print("🧪 Setting up test dataset...")
    test_dir, queries_csv, image_dir = create_test_dataset()
    
    print("\n🚀 Running text-to-image retrieval test...")
    
    # Import and run the retriever
    try:
        from text_to_image_retriever import TextToImageRetriever
        from clip_baseline.config import CLIPConfig
        
        # Use a fast configuration for testing
        config = CLIPConfig()
        config.model_name = "ViT-B/32"
        config.verbose = True
        
        # Initialize retriever
        retriever = TextToImageRetriever(config=config)
        
        # Load database
        print(f"📚 Loading database from: {image_dir}")
        retriever.load_database(str(image_dir))
        
        # Process queries
        output_csv = test_dir / "test_results.csv"
        print(f"🔍 Processing queries...")
        stats = retriever.process_csv_queries(str(queries_csv), str(output_csv), top_k=3)
        
        print(f"\n✅ Test completed successfully!")
        print(f"📊 Results saved to: {output_csv}")
        
        # Show sample results
        print(f"\n📋 Sample results:")
        results_df = pd.read_csv(output_csv)
        for query in results_df['query_description'].unique()[:2]:  # Show first 2 queries
            query_results = results_df[results_df['query_description'] == query]
            print(f"\nQuery: '{query}'")
            for _, result in query_results.iterrows():
                print(f"  {result['rank']}. {result['retrieved_filename']} (score: {result['similarity_score']:.3f})")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you have installed all required dependencies:")
        print("pip install torch torchvision transformers Pillow numpy scikit-learn tqdm openai-clip ftfy regex pandas")
        return False
    except Exception as e:
        print(f"❌ Error during test: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🧪 Text-to-Image Retriever Test")
    print("=" * 50)
    
    success = run_test()
    
    if success:
        print(f"\n🎉 Test completed successfully!")
        print(f"📁 Check the 'test_data' directory for results")
    else:
        print(f"\n❌ Test failed. Check the error messages above.")

