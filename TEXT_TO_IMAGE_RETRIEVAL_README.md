# Text-to-Image Retrieval with CLIP

This script uses CLIP (Contrastive Language-Image Pre-training) to find the most similar images from a database based on text descriptions. It takes a CSV file with image paths and descriptions, and for each description, retrieves the top-k most similar images from a database.

## Features

- **Text-to-Image Similarity**: Uses CLIP's text encoder to find images that match text descriptions
- **Flexible Database Support**: Works with image directories or pre-built database files
- **Configurable Retrieval**: Supports different CLIP models and similarity metrics
- **Batch Processing**: Processes multiple queries efficiently
- **Detailed Output**: Provides similarity scores and metadata for retrieved images

## Installation

Make sure you have the required dependencies installed:

```bash
pip install torch torchvision transformers Pillow numpy scikit-learn tqdm openai-clip ftfy regex pandas
```

## Usage

### Basic Usage

```bash
python text_to_image_retriever.py --input queries.csv --database ./image_database --output results.csv
```

### Advanced Usage

```bash
# Using a saved database file
python text_to_image_retriever.py --input queries.csv --database ./saved_database.npz --output results.csv --top_k 10

# Using a specific CLIP model
python text_to_image_retriever.py --input queries.csv --database ./images --output results.csv --model ViT-L/14 --top_k 5

# Using a configuration preset
python text_to_image_retriever.py --input queries.csv --database ./images --output results.csv --preset high_accuracy --top_k 20
```

## Input Format

The input CSV file should have the following columns:

| Column | Description |
|--------|-------------|
| `image_path` | Path to the query image (used for reference only) |
| `description` | Text description to search for |

### Example Input CSV

```csv
image_path,description
sample1.jpg,a red car parked on the street
sample2.jpg,a beautiful sunset over the ocean
sample3.jpg,a dog playing in the park
```

## Output Format

The output CSV file contains the following columns:

| Column | Description |
|--------|-------------|
| `query_image_path` | Original query image path |
| `utm_east` | UTM East coordinate of query image (extracted from filename) |
| `utm_north` | UTM North coordinate of query image (extracted from filename) |
| `reference_N_path` | Path to the N-th most similar image |
| `reference_N_distance` | Distance score (1.0 - similarity_score) |
| `reference_N_utm_east` | UTM East coordinate of reference image |
| `reference_N_utm_north` | UTM North coordinate of reference image |

**Note**: If CLIP fails to process a query or UTM coordinates cannot be extracted, `nan` values are used.

### Example Output CSV

```csv
query_image_path,utm_east,utm_north,reference_1_path,reference_1_distance,reference_1_utm_east,reference_1_utm_north,reference_2_path,reference_2_distance,reference_2_utm_east,reference_2_utm_north,reference_3_path,reference_3_distance,reference_3_utm_east,reference_3_utm_north
@0543256.96@4178906.70@image001.jpg,543256.96,4178906.7,database/car_001.jpg,0.1766,543260.15,4178911.85,database/vehicle_002.jpg,0.2109,543262.38,4178912.96,database/red_car_003.jpg,0.2346,543264.84,4178914.18
failed_query.jpg,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan
```

## Database Options

### Image Directory Database

Use a directory containing images:

```bash
python text_to_image_retriever.py --input queries.csv --database ./my_images --output results.csv
```

Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`

### Pre-built Database

Save a database for faster subsequent use:

```bash
# First time: build and save database
python text_to_image_retriever.py --input queries.csv --database ./my_images --output results.csv --save_database ./my_database.npz

# Subsequent uses: load saved database
python text_to_image_retriever.py --input queries.csv --database ./my_database.npz --output results.csv
```

## Configuration

### Configuration Presets

- `default`: Standard configuration for general use
- `vpr`: Optimized for Visual Place Recognition
- `fast`: Optimized for speed
- `high_accuracy`: Optimized for maximum accuracy

```bash
python text_to_image_retriever.py --input queries.csv --database ./images --output results.csv --preset high_accuracy
```

### Custom Configuration

Create a custom configuration file:

```json
{
  "model_name": "ViT-L/14",
  "similarity_metric": "cosine",
  "normalize_features": true,
  "default_top_k": 10,
  "min_similarity_threshold": 0.1,
  "verbose": true
}
```

Use the custom configuration:

```bash
python text_to_image_retriever.py --input queries.csv --database ./images --output results.csv --config my_config.json
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input` | Input CSV file path | Required |
| `--database` | Image database directory or .npz file | Required |
| `--output` | Output CSV file path | Required |
| `--top_k` | Number of top results to retrieve | 5 |
| `--config` | Configuration file path | None |
| `--preset` | Configuration preset name | None |
| `--model` | CLIP model variant | ViT-B/32 |
| `--device` | Device to use (cuda/cpu) | Auto-detect |
| `--save_database` | Path to save database | None |
| `--verbose` | Enable verbose output | False |

## Examples

### Example 1: Basic Text-to-Image Retrieval

```bash
python text_to_image_retriever.py \
  --input sample_queries.csv \
  --database ./image_database \
  --output retrieval_results.csv \
  --top_k 5
```

### Example 2: High-Accuracy Retrieval with Large Model

```bash
python text_to_image_retriever.py \
  --input queries.csv \
  --database ./images \
  --output results.csv \
  --model ViT-L/14 \
  --preset high_accuracy \
  --top_k 10 \
  --verbose
```

### Example 3: Fast Processing with Pre-built Database

```bash
# Build database once
python text_to_image_retriever.py \
  --input queries.csv \
  --database ./large_image_collection \
  --output results.csv \
  --save_database ./large_database.npz \
  --preset fast

# Use saved database for faster processing
python text_to_image_retriever.py \
  --input new_queries.csv \
  --database ./large_database.npz \
  --output new_results.csv
```

## Performance Tips

1. **Use Pre-built Databases**: For large image collections, build and save the database once, then reuse it
2. **Choose Appropriate Model**: 
   - `ViT-B/32`: Good balance of speed and accuracy
   - `ViT-L/14`: Higher accuracy but slower
3. **GPU Acceleration**: Use CUDA if available for faster processing
4. **Batch Processing**: The script processes queries efficiently in batches

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Use a smaller model or process in smaller batches
2. **Slow Processing**: Use `--preset fast` or a smaller model
3. **Low Similarity Scores**: Try `--preset high_accuracy` or adjust similarity threshold

### Error Messages

- `No images found in directory`: Check that the database path contains supported image formats
- `CSV missing required columns`: Ensure input CSV has `image_path` and `description` columns
- `Error processing query`: Check that descriptions are valid text strings

## Technical Details

- **Model**: Uses OpenAI CLIP models (ViT-B/32 by default)
- **Similarity**: Cosine similarity between text and image embeddings
- **Normalization**: Features are L2-normalized for better similarity computation
- **Device**: Automatically detects and uses GPU if available

## License

This script is part of the CLIP baseline project and follows the same license terms.
