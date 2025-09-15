# CLIP Baseline for Text-based Visual Place Recognition

This directory contains a CLIP-based baseline system for text-to-image retrieval, which can be used for visual place recognition tasks.

## Features

- **CLIP Integration**: Uses OpenAI's CLIP model for zero-shot image-text matching
- **Batch Processing**: Efficiently processes large image datasets
- **Database Persistence**: Save and load pre-computed image features
- **Flexible Search**: Query with natural language descriptions
- **Multiple Models**: Support for different CLIP model variants
- **GPU/CPU Support**: Automatic device detection and optimization

## Installation

1. **Activate your conda environment:**
   ```bash
   conda activate text_vpr
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Command Line Interface

The main script `clip_retriever.py` provides a command-line interface:

```bash
# Basic usage - build database and search
python clip_retriever.py --image_dir /path/to/images --text_prompt "modern building with glass windows" --top_k 5

# Save database for future use
python clip_retriever.py --image_dir /path/to/images --text_prompt "red car" --save_db database.npz

# Load existing database
python clip_retriever.py --load_db database.npz --text_prompt "outdoor landscape" --top_k 10

# Save results to JSON file
python clip_retriever.py --image_dir /path/to/images --text_prompt "people walking" --output_file results.json
```

### Command Line Arguments

- `--image_dir`: Directory containing images to index
- `--text_prompt`: Text description to search for
- `--top_k`: Number of top results to return (default: 5)
- `--model`: CLIP model variant (default: "ViT-B/32")
- `--device`: Device to use ("cuda" or "cpu", auto-detect if not specified)
- `--save_db`: Path to save the image database
- `--load_db`: Path to load existing database
- `--output_file`: Path to save results as JSON

### Programmatic Usage

```python
from clip_retriever import CLIPRetriever

# Initialize retriever
retriever = CLIPRetriever(model_name="ViT-B/32")

# Build image database
retriever.build_image_database("/path/to/images")

# Search for images
results = retriever.search("modern building with glass windows", top_k=5)

# Save database
retriever.save_database("database.npz")

# Load database later
retriever.load_database("database.npz")
```

### Demo Script

Run the demo script to see the system in action:

```bash
python demo.py
```

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)
- WebP (.webp)

## CLIP Model Variants

Available CLIP models:
- `ViT-B/32`: Fastest, good performance (default)
- `ViT-B/16`: Better performance, slower
- `ViT-L/14`: Best performance, slowest
- `RN50`: ResNet-based model

## Performance Tips

1. **GPU Usage**: Use CUDA for significantly faster processing
2. **Database Persistence**: Save and reuse image features to avoid re-encoding
3. **Batch Size**: The system automatically processes images in batches
4. **Model Selection**: Choose model based on speed vs. accuracy trade-off

## Example Use Cases

### Visual Place Recognition
```bash
python clip_retriever.py --image_dir places_dataset --text_prompt "busy intersection with traffic lights and crosswalk" --top_k 10
```

### Object Search
```bash
python clip_retriever.py --image_dir object_dataset --text_prompt "red sports car with black wheels" --top_k 5
```

### Scene Description
```bash
python clip_retriever.py --image_dir scene_dataset --text_prompt "modern office building with glass facade and concrete structure" --top_k 8
```

## Output Format

Results include:
- Rank position
- Image file path
- Filename
- Similarity score (0-1, higher is better)
- Image metadata (size, format, etc.)

## Troubleshooting

1. **CUDA Out of Memory**: Use smaller CLIP model or process fewer images at once
2. **Slow Processing**: Ensure GPU is being used, consider using smaller model
3. **No Images Found**: Check image directory path and supported formats
4. **Import Errors**: Ensure all requirements are installed in your conda environment

## Dependencies

- PyTorch
- CLIP (OpenAI)
- PIL/Pillow
- NumPy
- scikit-learn
- tqdm
- ftfy
- regex
