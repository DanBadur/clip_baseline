#!/usr/bin/env python3
"""
Simple demo script for CLIP-based image retrieval
"""

from clip_retriever import CLIPRetriever
import os

def main():
    # Example usage
    print("CLIP Image Retrieval Demo")
    print("=" * 50)
    
    # Initialize retriever (will auto-detect GPU/CPU)
    retriever = CLIPRetriever(model_name="ViT-B/32")
    
    # Example image directory (modify this path)
    image_dir = "images"  # Change this to your image directory
    
    if not os.path.exists(image_dir):
        print(f"Image directory '{image_dir}' not found!")
        print("Please create an 'images' folder with some images or modify the path in this script.")
        return
    
    # Build image database
    print(f"\nBuilding database from: {image_dir}")
    retriever.build_image_database(image_dir)
    
    # Example search queries
    queries = [
        "a red car",
        "building with windows",
        "outdoor landscape",
        "people walking",
        "modern architecture"
    ]
    
    # Perform searches
    for query in queries:
        print(f"\nSearching for: '{query}'")
        print("-" * 40)
        
        results = retriever.search(query, top_k=3)
        
        for result in results:
            print(f"{result['rank']}. {result['filename']} (Score: {result['similarity_score']:.4f})")
    
    # Save database for future use
    retriever.save_database("image_database.npz")
    print(f"\nDatabase saved to 'image_database.npz'")

if __name__ == "__main__":
    main()
