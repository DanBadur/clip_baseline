#!/usr/bin/env python3
"""
Example usage of CLIP retriever for text-based image search
"""

from clip_retriever import CLIPRetriever
import os

def main():
    print("CLIP Text-to-Image Retrieval Example")
    print("=" * 50)
    
    # Initialize the CLIP retriever
    print("Initializing CLIP retriever...")
    retriever = CLIPRetriever(model_name="ViT-B/32")
    
    # Example: Create a simple images directory if it doesn't exist
    images_dir = "sample_images"
    if not os.path.exists(images_dir):
        print(f"\nCreating sample images directory: {images_dir}")
        os.makedirs(images_dir, exist_ok=True)
        print("Please add some images to the 'sample_images' folder and run this script again.")
        print("Supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .webp")
        return
    
    # Check if there are images in the directory
    image_files = [f for f in os.listdir(images_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'))]
    
    if not image_files:
        print(f"\nNo images found in {images_dir}")
        print("Please add some images and run this script again.")
        return
    
    print(f"\nFound {len(image_files)} images in {images_dir}")
    
    # Build the image database
    print("\nBuilding image feature database...")
    retriever.build_image_database(images_dir)
    
    # Example search queries
    example_queries = [
        "modern building",
        "outdoor scene",
        "people",
        "vehicle",
        "nature landscape"
    ]
    
    print(f"\nPerforming searches with example queries...")
    print("-" * 50)
    
    for query in example_queries:
        print(f"\nSearching for: '{query}'")
        results = retriever.search(query, top_k=3)
        
        for result in results:
            print(f"  {result['rank']}. {result['filename']} (Score: {result['similarity_score']:.4f})")
    
    # Save the database for future use
    db_path = "image_database.npz"
    retriever.save_database(db_path)
    print(f"\nDatabase saved to: {db_path}")
    
    # Demonstrate loading the database
    print(f"\nLoading database from: {db_path}")
    new_retriever = CLIPRetriever(model_name="ViT-B/32")
    new_retriever.load_database(db_path)
    
    # Test a new query on loaded database
    test_query = "architecture"
    print(f"\nTesting loaded database with query: '{test_query}'")
    results = new_retriever.search(test_query, top_k=2)
    
    for result in results:
        print(f"  {result['rank']}. {result['filename']} (Score: {result['similarity_score']:.4f})")
    
    print(f"\n✅ Example completed successfully!")
    print(f"\nTo use with your own images:")
    print(f"1. Place images in the '{images_dir}' folder")
    print(f"2. Run: python clip_retriever.py --image_dir {images_dir} --text_prompt 'your description' --top_k 5")
    print(f"3. Or use the programmatic interface as shown in this example")

if __name__ == "__main__":
    main()
