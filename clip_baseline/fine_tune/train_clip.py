#!/usr/bin/env python3
"""
Main training script for CLIP fine-tuning
Demonstrates usage of all components for improving top-k image retrieval
"""

import argparse
import json
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import logging
from datetime import datetime

from fine_tune_trainer import CLIPFineTuneTrainer
from fine_tune_dataset import (
    CLIPFineTuneDataset, 
    CLIPHardNegativeDataset, 
    CLIPTripletDataset,
    create_dataloader
)
from fine_tune_utils import CLIPDataUtilities


def setup_logging(config):
    """Setup logging configuration"""
    log_dir = Path(config['logging_config']['log_dir'])
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    
    logging.basicConfig(
        level=getattr(logging, config['logging_config']['log_level']),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def create_datasets(config):
    """Create training and validation datasets based on configuration"""
    dataset_config = config['dataset_config']
    augmentation_config = config['augmentation_config']
    
    dataset_type = dataset_config['dataset_type']
    
    # Create training dataset
    if dataset_type == "standard":
        train_dataset = CLIPFineTuneDataset(
            data_path=dataset_config['train_data_path'],
            clip_model_name=config['model_config']['model_name'],
            split="train",
            max_length=dataset_config['max_length'],
            use_augmentation=dataset_config['use_augmentation'],
            augmentation_config=augmentation_config
        )
    elif dataset_type == "hard_negative":
        train_dataset = CLIPHardNegativeDataset(
            data_path=dataset_config['train_data_path'],
            clip_model_name=config['model_config']['model_name'],
            split="train",
            max_length=dataset_config['max_length'],
            negative_mining_strategy=dataset_config['negative_mining_strategy'],
            num_negatives=config['loss_config']['num_hard_negatives'],
            temperature=config['loss_config']['temperature']
        )
    elif dataset_type == "triplet":
        train_dataset = CLIPTripletDataset(
            data_path=dataset_config['train_data_path'],
            clip_model_name=config['model_config']['model_name'],
            split="train",
            max_length=dataset_config['max_length'],
            triplet_strategy=dataset_config['triplet_strategy'],
            margin=config['loss_config']['margin']
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Create validation dataset (always standard for evaluation)
    val_dataset = None
    if os.path.exists(dataset_config['val_data_path']):
        val_dataset = CLIPFineTuneDataset(
            data_path=dataset_config['val_data_path'],
            clip_model_name=config['model_config']['model_name'],
            split="val",
            max_length=dataset_config['max_length'],
            use_augmentation=False  # No augmentation for validation
        )
    
    return train_dataset, val_dataset


def create_dataloaders(train_dataset, val_dataset, config):
    """Create data loaders with appropriate collators"""
    dataset_type = config['dataset_config']['dataset_type']
    batch_size = config['training_config']['batch_size']
    num_workers = config['hardware_config']['num_workers']
    
    # Create training dataloader
    train_loader = create_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        dataset_type=dataset_type,
        shuffle=True,
        num_workers=num_workers
    )
    
    # Create validation dataloader
    val_loader = None
    if val_dataset:
        val_loader = create_dataloader(
            dataset=val_dataset,
            batch_size=batch_size,
            dataset_type="standard",  # Always standard for validation
            shuffle=False,
            num_workers=num_workers
        )
    
    return train_loader, val_loader


def balance_dataset(dataset, config):
    """Balance dataset if specified in configuration"""
    balance_config = config['data_balancing_config']
    
    if balance_config['balance_strategy'] == 'none':
        return dataset
    
    # Extract image-caption pairs for balancing
    image_caption_pairs = []
    for i in range(len(dataset)):
        sample = dataset[i]
        image_caption_pairs.append({
            'image_path': sample['image_path'],
            'caption': sample['caption'],
            'category': sample.get('category', 'unknown')
        })
    
    # Apply balancing
    balanced_pairs = CLIPDataUtilities.create_balanced_dataset(
        image_caption_pairs,
        balance_strategy=balance_config['balance_strategy']
    )
    
    print(f"Balanced dataset: {len(balanced_pairs)} samples")
    return balanced_pairs


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="CLIP Fine-tuning Training")
    parser.add_argument("--config", type=str, default="fine_tune_config.json",
                       help="Path to training configuration file")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--data_dir", type=str, default=None,
                       help="Override data directory from config")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Output directory for trained models")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Override data directory if specified
    if args.data_dir:
        config['dataset_config']['train_data_path'] = os.path.join(args.data_dir, 'train')
        config['dataset_config']['val_data_path'] = os.path.join(args.data_dir, 'val')
        config['dataset_config']['test_data_path'] = os.path.join(args.data_dir, 'test')
    
    # Setup logging
    logger = setup_logging(config)
    logger.info("Starting CLIP fine-tuning training")
    logger.info(f"Configuration: {args.config}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset, val_dataset = create_datasets(config)
    
    # Balance dataset if specified
    if config['data_balancing_config']['balance_strategy'] != 'none':
        logger.info("Balancing dataset...")
        balanced_data = balance_dataset(train_dataset, config)
        # Note: In a real implementation, you'd need to recreate the dataset with balanced data
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, config)
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = CLIPFineTuneTrainer(args.config)
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    logger.info("Starting training...")
    training_history = trainer.train(train_dataset, val_dataset)
    
    # Save final model
    final_model_path = output_dir / "final_model.pt"
    trainer.save_final_model(str(final_model_path))
    
    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    logger.info("Training completed successfully!")
    logger.info(f"Final model saved to: {final_model_path}")
    logger.info(f"Training history saved to: {history_path}")


def create_sample_data_structure():
    """Create sample data structure for demonstration"""
    print("Creating sample data structure...")
    
    # Create directories
    data_dir = Path("sample_data")
    data_dir.mkdir(exist_ok=True)
    
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        split_dir.mkdir(exist_ok=True)
        
        # Create sample annotations
        annotations = {
            "splits": {
                split: [
                    {
                        "image_path": f"images/{split}_001.jpg",
                        "caption": f"A sample {split} image with description",
                        "id": 0
                    },
                    {
                        "image_path": f"images/{split}_002.jpg",
                        "caption": f"Another {split} image example",
                        "id": 1
                    }
                ]
            }
        }
        
        with open(split_dir / "annotations.json", 'w') as f:
            json.dump(annotations, f, indent=2)
    
    print("Sample data structure created in 'sample_data' directory")
    print("You can now run training with: python train_clip.py --data_dir sample_data")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # No arguments provided, show help and create sample data
        print("CLIP Fine-tuning Training Script")
        print("=" * 40)
        print("\nUsage:")
        print("python train_clip.py --config fine_tune_config.json --data_dir /path/to/data")
        print("\nArguments:")
        print("--config: Path to training configuration file")
        print("--resume: Path to checkpoint to resume from (optional)")
        print("--data_dir: Override data directory from config")
        print("--output_dir: Output directory for trained models")
        
        print("\nCreating sample data structure...")
        create_sample_data_structure()
        
        print("\nExample training command:")
        print("python train_clip.py --config fine_tune_config.json --data_dir sample_data")
    else:
        main()

