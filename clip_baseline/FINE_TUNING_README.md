# CLIP Fine-tuning for Improved Top-K Image Retrieval

This directory contains comprehensive scripts for fine-tuning CLIP models to achieve better top-k image retrieval performance for text descriptions. The system implements advanced training strategies including hard negative mining, triplet learning, and various loss functions.

## 🎯 Key Features

- **Multiple Loss Functions**: Contrastive, Triplet, and InfoNCE losses
- **Hard Negative Mining**: Automatically finds challenging negative examples
- **Triplet Learning**: Anchor-positive-negative training strategy
- **Data Augmentation**: Image and text augmentation for robustness
- **Flexible Training**: Support for different dataset types and configurations
- **Comprehensive Evaluation**: Multiple retrieval metrics (Recall@K, mAP, NDCG)

## 📁 File Structure

```
clip_baseline/
├── fine_tune_trainer.py      # Main training class with all loss functions
├── fine_tune_dataset.py      # Specialized datasets for different training strategies
├── fine_tune_utils.py        # Utility functions for augmentation and model management
├── fine_tune_config.json     # Comprehensive training configuration
├── train_clip.py             # Main training script
└── FINE_TUNING_README.md     # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision clip transformers pillow tqdm numpy
```

### 2. Prepare Your Data

Organize your data in one of these formats:

**Option A: JSON annotations file**
```json
{
  "splits": {
    "train": [
      {
        "image_path": "path/to/image1.jpg",
        "caption": "A modern building with glass windows",
        "id": 0
      }
    ],
    "val": [...],
    "test": [...]
  }
}
```

**Option B: Directory structure with sidecar files**
```
data/
├── train/
│   ├── image1.jpg
│   ├── image1.txt          # Caption file
│   └── image1.json         # Metadata file
├── val/
└── test/
```

### 3. Run Training

```bash
# Basic training
python train_clip.py --config fine_tune_config.json --data_dir /path/to/data

# Resume from checkpoint
python train_clip.py --config fine_tune_config.json --data_dir /path/to/data --resume checkpoints/checkpoint_epoch_5.pt

# Custom output directory
python train_clip.py --config fine_tune_config.json --data_dir /path/to/data --output_dir my_models
```

## 🎯 Training Data Recommendations

### For Top-K Retrieval Improvement

1. **High-Quality Image-Caption Pairs**
   - Use detailed, descriptive captions
   - Ensure captions accurately describe visual content
   - Include spatial relationships and object attributes

2. **Diverse Content**
   - Cover various scenes, objects, and contexts
   - Include different lighting conditions and viewpoints
   - Balance between simple and complex descriptions

3. **Hard Negative Examples**
   - Include visually similar images with different descriptions
   - Add images with partial caption matches
   - Use images that could be confused with each other

### Example Training Data

```json
{
  "image_path": "buildings/001.jpg",
  "caption": "Modern glass skyscraper with reflective windows, blue sky background, urban cityscape",
  "category": "architecture"
}
```

## 🧠 Loss Function Selection

### 1. Contrastive Loss with Hard Negatives (Recommended)
- **Best for**: General retrieval improvement
- **Pros**: Automatically finds challenging examples, stable training
- **Use when**: You want to improve overall retrieval quality

```json
{
  "loss_config": {
    "loss_type": "contrastive",
    "hard_negative_mining": true,
    "temperature": 0.07,
    "margin": 0.5
  }
}
```

### 2. Triplet Loss
- **Best for**: Learning fine-grained feature representations
- **Pros**: Explicit positive-negative relationships, good for ranking
- **Use when**: You need precise ranking of similar images

```json
{
  "loss_config": {
    "loss_type": "triplet",
    "margin": 0.5
  }
}
```

### 3. InfoNCE Loss
- **Best for**: Large-scale contrastive learning
- **Pros**: Information-theoretic approach, good temperature scaling
- **Use when**: Working with large datasets and want theoretical soundness

```json
{
  "loss_config": {
    "loss_type": "info_nce",
    "temperature": 0.07
  }
}
```

## ⚙️ Configuration Guide

### Model Configuration
```json
{
  "model_config": {
    "model_name": "ViT-B/32",        // CLIP model variant
    "freeze_image_encoder": false,   // Whether to freeze image encoder
    "freeze_text_encoder": false,    // Whether to freeze text encoder
    "freeze_projections": false      // Whether to freeze projection layers
  }
}
```

### Training Configuration
```json
{
  "training_config": {
    "epochs": 20,                    // Number of training epochs
    "batch_size": 32,                // Batch size
    "learning_rate": 1e-5,           // Learning rate
    "patience": 5,                   // Early stopping patience
    "eval_every": 1                  // Validation frequency
  }
}
```

### Dataset Configuration
```json
{
  "dataset_config": {
    "dataset_type": "hard_negative", // "standard", "hard_negative", or "triplet"
    "use_augmentation": true,        // Enable data augmentation
    "negative_mining_strategy": "semantic"  // "semantic" or "random"
  }
}
```

## 🔧 Advanced Training Strategies

### 1. Hard Negative Mining
Automatically finds challenging negative examples during training:

```python
from fine_tune_dataset import CLIPHardNegativeDataset

dataset = CLIPHardNegativeDataset(
    data_path="data/train",
    negative_mining_strategy="semantic",
    num_negatives=3,
    temperature=0.07
)
```

### 2. Triplet Generation
Creates anchor-positive-negative triplets for better feature learning:

```python
from fine_tune_dataset import CLIPTripletDataset

dataset = CLIPTripletDataset(
    data_path="data/train",
    triplet_strategy="hardest",  # "hardest" or "semi_hard"
    margin=0.5
)
```

### 3. Data Augmentation
Enhances training data with image and text variations:

```json
{
  "augmentation_config": {
    "color_jitter": 0.4,
    "random_horizontal_flip": 0.5,
    "random_rotation": 10,
    "augment_captions": true,
    "max_caption_variations": 3
  }
}
```

## 📊 Evaluation Metrics

The system automatically computes:

- **Recall@K**: Percentage of relevant images in top-K results
- **mAP**: Mean Average Precision across all queries
- **NDCG**: Normalized Discounted Cumulative Gain
- **Training Loss**: Various loss function values

### Monitoring Training

```bash
# Check logs
tail -f logs/training_*.log

# View training history
cat outputs/training_history.json
```

## 🎨 Customization Examples

### Custom Loss Function
```python
class CustomLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, image_features, text_features, labels):
        # Your custom loss implementation
        pass

# Use in trainer
trainer.criterion = CustomLoss(temperature=0.1)
```

### Custom Dataset
```python
class CustomDataset(CLIPFineTuneDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add custom functionality
    
    def __getitem__(self, idx):
        # Custom data loading logic
        pass
```

## 🚨 Common Issues and Solutions

### 1. Out of Memory
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision training

```json
{
  "training_config": {
    "batch_size": 16
  },
  "hardware_config": {
    "gradient_accumulation_steps": 2,
    "mixed_precision": true
  }
}
```

### 2. Slow Training
- Use GPU if available
- Reduce number of workers
- Use smaller CLIP model variant

### 3. Poor Convergence
- Check learning rate
- Verify data quality
- Try different loss functions
- Adjust temperature and margin parameters

## 📈 Performance Tips

1. **Start Small**: Begin with a small dataset to verify setup
2. **Monitor Metrics**: Watch validation metrics for overfitting
3. **Experiment**: Try different loss functions and configurations
4. **Data Quality**: Ensure high-quality image-caption pairs
5. **Regularization**: Use data augmentation and early stopping

## 🔍 Example Use Cases

### Visual Place Recognition
```bash
# Train on place dataset
python train_clip.py \
  --config fine_tune_config.json \
  --data_dir places_dataset \
  --output_dir place_recognition_model
```

### Object Retrieval
```bash
# Train on object dataset
python train_clip.py \
  --config fine_tune_config.json \
  --data_dir objects_dataset \
  --output_dir object_retrieval_model
```

### Scene Understanding
```bash
# Train on scene dataset
python train_clip.py \
  --config fine_tune_config.json \
  --data_dir scenes_dataset \
  --output_dir scene_understanding_model
```

## 📚 Additional Resources

- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [Contrastive Learning Tutorial](https://pytorch.org/tutorials/beginner/contrastive_learning.html)
- [Hard Negative Mining Strategies](https://arxiv.org/abs/1703.07737)

## 🤝 Contributing

Feel free to contribute improvements:
- Add new loss functions
- Implement additional dataset types
- Enhance evaluation metrics
- Optimize training strategies

## 📄 License

This project follows the same license as the original CLIP model.

---

**Happy fine-tuning! 🚀**

For questions or issues, please check the logs and configuration files first, then refer to the troubleshooting section above.


