#!/usr/bin/env python3
"""
CLIP Fine-tuning Dataset Classes
Specialized datasets for training CLIP models with hard negative mining and data augmentation
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import random
import numpy as np
from tqdm import tqdm
import clip

from fine_tune_utils import CLIPDataAugmentation, CLIPDataUtilities


class CLIPFineTuneDataset(Dataset):
    """Base dataset class for CLIP fine-tuning"""
    
    def __init__(self, data_path: str, clip_model_name: str = "ViT-B/32", 
                 split: str = "train", max_length: int = 77, 
                 use_augmentation: bool = True, augmentation_config: Optional[Dict] = None):
        """
        Initialize the dataset
        
        Args:
            data_path: Path to data directory or JSON file
            clip_model_name: CLIP model name for preprocessing
            split: Dataset split ('train', 'val', 'test')
            max_length: Maximum text length for CLIP tokenizer
            use_augmentation: Whether to use data augmentation
            augmentation_config: Configuration for data augmentation
        """
        self.data_path = data_path
        self.split = split
        self.max_length = max_length
        self.use_augmentation = use_augmentation
        
        # Load CLIP preprocessing
        self.clip_model, self.preprocess = clip.load(clip_model_name, device="cpu")
        self.tokenizer = clip.tokenize
        
        # Setup augmentation
        if use_augmentation and augmentation_config:
            self.augmenter = CLIPDataAugmentation(augmentation_config)
        else:
            self.augmenter = None
        
        # Load data
        self.data = self._load_data()
        
        # Setup transforms
        self.transforms = self._setup_transforms()
        
        print(f"Loaded {len(self.data)} samples for {split} split")
    
    def _load_data(self) -> List[Dict]:
        """Load dataset from file or directory"""
        if self.data_path.endswith('.json'):
            return self._load_from_json()
        elif os.path.isdir(self.data_path):
            return self._load_from_directory()
        else:
            raise ValueError(f"Unsupported data path: {self.data_path}")
    
    def _load_from_json(self) -> List[Dict]:
        """Load data from JSON file"""
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        # Filter by split if available
        if isinstance(data, dict) and 'splits' in data:
            if self.split in data['splits']:
                return data['splits'][self.split]
            else:
                raise ValueError(f"Split '{self.split}' not found in data")
        else:
            return data
    
    def _load_from_directory(self) -> List[Dict]:
        """Load data from directory structure"""
        data = []
        data_dir = Path(self.data_path)
        
        # Look for common directory structures
        if (data_dir / "annotations.json").exists():
            return self._load_from_json()
        
        # Assume images with captions in filename or sidecar files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        for img_path in data_dir.rglob("*"):
            if img_path.suffix.lower() in image_extensions:
                # Try to find caption
                caption = self._extract_caption(img_path)
                if caption:
                    data.append({
                        'image_path': str(img_path),
                        'caption': caption,
                        'id': len(data)
                    })
        
        return data
    
    def _extract_caption(self, img_path: Path) -> Optional[str]:
        """Extract caption from image filename or sidecar file"""
        # Try sidecar caption file
        caption_file = img_path.with_suffix('.txt')
        if caption_file.exists():
            with open(caption_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        
        # Try JSON sidecar file
        json_file = img_path.with_suffix('.json')
        if json_file.exists():
            try:
                with open(json_file, 'r') as f:
                    metadata = json.load(f)
                    if 'caption' in metadata:
                        return metadata['caption']
                    elif 'description' in metadata:
                        return metadata['description']
            except:
                pass
        
        # Extract from filename (remove extension and replace underscores)
        caption = img_path.stem.replace('_', ' ').replace('-', ' ')
        if len(caption) > 10:  # Reasonable caption length
            return caption
        
        return None
    
    def _setup_transforms(self) -> transforms.Compose:
        """Setup image transforms"""
        transform_list = []
        
        if self.use_augmentation and self.augmenter:
            # Add augmentation transforms
            transform_list.extend(self.augmenter.transforms.transforms)
        
        # Add CLIP preprocessing
        transform_list.append(self.preprocess)
        
        return transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample"""
        sample = self.data[idx]
        
        # Load and preprocess image
        image = Image.open(sample['image_path']).convert('RGB')
        image = self.transforms(image)
        
        # Tokenize text
        text = sample['caption']
        text_tokens = self.tokenizer(text, truncate=True, max_length=self.max_length)
        
        return {
            'images': image,
            'texts': text_tokens,
            'labels': torch.tensor(sample['id'], dtype=torch.long),
            'image_path': sample['image_path'],
            'caption': text
        }


class CLIPHardNegativeDataset(Dataset):
    """Dataset with hard negative mining for better CLIP training"""
    
    def __init__(self, data_path: str, clip_model_name: str = "ViT-B/32", 
                 split: str = "train", max_length: int = 77, 
                 negative_mining_strategy: str = "semantic", 
                 num_negatives: int = 3, temperature: float = 0.07):
        """
        Initialize hard negative dataset
        
        Args:
            data_path: Path to data
            clip_model_name: CLIP model name
            split: Dataset split
            max_length: Maximum text length
            negative_mining_strategy: Strategy for finding hard negatives
            num_negatives: Number of hard negatives per positive
            temperature: Temperature for similarity computation
        """
        self.data_path = data_path
        self.split = split
        self.max_length = max_length
        self.negative_mining_strategy = negative_mining_strategy
        self.num_negatives = num_negatives
        self.temperature = temperature
        
        # Load base dataset
        self.base_dataset = CLIPFineTuneDataset(
            data_path, clip_model_name, split, max_length, 
            use_augmentation=False  # We'll handle augmentation differently
        )
        
        # Precompute hard negatives
        self.hard_negatives = self._precompute_hard_negatives()
        
        print(f"Precomputed hard negatives for {len(self.hard_negatives)} samples")
    
    def _precompute_hard_negatives(self) -> List[List[int]]:
        """Precompute hard negative indices for each sample"""
        print("Computing hard negatives...")
        
        # Get all features first
        all_features = []
        all_captions = []
        
        for i in tqdm(range(len(self.base_dataset)), desc="Computing features"):
            sample = self.base_dataset[i]
            with torch.no_grad():
                # Get image features
                image_features = self.base_dataset.clip_model.encode_image(
                    sample['images'].unsqueeze(0)
                )
                # Get text features
                text_features = self.base_dataset.clip_model.encode_text(
                    sample['texts'].unsqueeze(0)
                )
                
                # Average image and text features
                combined_features = (image_features + text_features) / 2
                all_features.append(combined_features.squeeze())
                all_captions.append(sample['caption'])
        
        # Stack all features
        all_features = torch.stack(all_features)
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(all_features, all_features.T) / self.temperature
        
        # Find hard negatives for each sample
        hard_negatives = []
        
        for i in range(len(all_features)):
            # Get similarities to other samples
            similarities = similarity_matrix[i]
            
            # Remove self-similarity
            similarities[i] = -float('inf')
            
            # Find hardest negatives (highest similarity among non-matching)
            if self.negative_mining_strategy == "semantic":
                # Use semantic similarity
                hard_negative_indices = torch.topk(similarities, k=self.num_negatives)[1]
            elif self.negative_mining_strategy == "random":
                # Random negatives
                other_indices = list(range(len(all_features)))
                other_indices.remove(i)
                hard_negative_indices = torch.tensor(
                    random.sample(other_indices, min(self.num_negatives, len(other_indices)))
                )
            else:
                raise ValueError(f"Unknown negative mining strategy: {self.negative_mining_strategy}")
            
            hard_negatives.append(hard_negative_indices.tolist())
        
        return hard_negatives
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get sample with hard negatives"""
        # Get positive sample
        positive_sample = self.base_dataset[idx]
        
        # Get hard negatives
        negative_indices = self.hard_negatives[idx]
        
        # Create negative samples
        negative_images = []
        negative_texts = []
        
        for neg_idx in negative_indices:
            neg_sample = self.base_dataset[neg_idx]
            negative_images.append(neg_sample['images'])
            negative_texts.append(neg_sample['texts'])
        
        # Stack negatives
        negative_images = torch.stack(negative_images)
        negative_texts = torch.stack(negative_texts)
        
        return {
            'positive_image': positive_sample['images'],
            'positive_text': positive_sample['texts'],
            'negative_images': negative_images,
            'negative_texts': negative_texts,
            'label': positive_sample['labels'],
            'image_path': positive_sample['image_path'],
            'caption': positive_sample['caption']
        }


class CLIPTripletDataset(Dataset):
    """Dataset for triplet-based CLIP training"""
    
    def __init__(self, data_path: str, clip_model_name: str = "ViT-B/32", 
                 split: str = "train", max_length: int = 77, 
                 triplet_strategy: str = "hardest", margin: float = 0.5):
        """
        Initialize triplet dataset
        
        Args:
            data_path: Path to data
            clip_model_name: CLIP model name
            split: Dataset split
            max_length: Maximum text length
            triplet_strategy: Strategy for selecting triplets
            margin: Margin for triplet loss
        """
        self.data_path = data_path
        self.split = split
        self.max_length = max_length
        self.triplet_strategy = triplet_strategy
        self.margin = margin
        
        # Load base dataset
        self.base_dataset = CLIPFineTuneDataset(
            data_path, clip_model_name, split, max_length, 
            use_augmentation=False
        )
        
        # Precompute triplets
        self.triplets = self._precompute_triplets()
        
        print(f"Precomputed {len(self.triplets)} triplets")
    
    def _precompute_triplets(self) -> List[Dict]:
        """Precompute triplets for training"""
        print("Computing triplets...")
        
        # Get all features
        all_features = []
        all_captions = []
        
        for i in tqdm(range(len(self.base_dataset)), desc="Computing features"):
            sample = self.base_dataset[i]
            with torch.no_grad():
                image_features = self.base_dataset.clip_model.encode_image(
                    sample['images'].unsqueeze(0)
                )
                text_features = self.base_dataset.clip_model.encode_text(
                    sample['texts'].unsqueeze(0)
                )
                combined_features = (image_features + text_features) / 2
                all_features.append(combined_features.squeeze())
                all_captions.append(sample['caption'])
        
        all_features = torch.stack(all_features)
        
        # Compute pairwise distances
        distances = torch.cdist(all_features, all_features)
        
        # Create triplets
        triplets = []
        
        for i in range(len(all_features)):
            # Find positive pairs (similar content)
            positive_candidates = []
            for j in range(len(all_features)):
                if i != j:
                    # Simple similarity based on caption overlap
                    caption_i = all_captions[i].lower().split()
                    caption_j = all_captions[j].lower().split()
                    overlap = len(set(caption_i) & set(caption_j))
                    if overlap >= 2:  # At least 2 words overlap
                        positive_candidates.append((j, distances[i, j]))
            
            if len(positive_candidates) == 0:
                continue
            
            # Sort by distance (closest first for hardest positive)
            positive_candidates.sort(key=lambda x: x[1])
            
            # Find negative pairs (dissimilar content)
            negative_candidates = []
            for j in range(len(all_features)):
                if i != j:
                    caption_i = all_captions[i].lower().split()
                    caption_j = all_captions[j].lower().split()
                    overlap = len(set(caption_i) & set(caption_j))
                    if overlap < 2:  # Less than 2 words overlap
                        negative_candidates.append((j, distances[i, j]))
            
            if len(negative_candidates) == 0:
                continue
            
            # Sort by distance (closest first for hardest negative)
            negative_candidates.sort(key=lambda x: x[1])
            
            # Create triplets
            if self.triplet_strategy == "hardest":
                # Hardest positive and hardest negative
                anchor_idx = i
                positive_idx = positive_candidates[-1][0]  # Furthest positive
                negative_idx = negative_candidates[0][0]   # Closest negative
                
                # Check if triplet satisfies margin
                if distances[anchor_idx, positive_idx] + self.margin < distances[anchor_idx, negative_idx]:
                    triplets.append({
                        'anchor': anchor_idx,
                        'positive': positive_idx,
                        'negative': negative_idx
                    })
            
            elif self.triplet_strategy == "semi_hard":
                # Semi-hard negative (closer than positive but satisfies margin)
                anchor_idx = i
                positive_idx = positive_candidates[0][0]  # Closest positive
                
                for neg_idx, neg_dist in negative_candidates:
                    if (distances[anchor_idx, positive_idx] < neg_dist < 
                        distances[anchor_idx, positive_idx] + self.margin):
                        triplets.append({
                            'anchor': anchor_idx,
                            'positive': positive_idx,
                            'negative': neg_idx
                        })
                        break
        
        return triplets
    
    def __len__(self) -> int:
        return len(self.triplets)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a triplet sample"""
        triplet = self.triplets[idx]
        
        # Get samples
        anchor_sample = self.base_dataset[triplet['anchor']]
        positive_sample = self.base_dataset[triplet['positive']]
        negative_sample = self.base_dataset[triplet['negative']]
        
        return {
            'anchor_image': anchor_sample['images'],
            'anchor_text': anchor_sample['texts'],
            'positive_image': positive_sample['images'],
            'positive_text': positive_sample['texts'],
            'negative_image': negative_sample['images'],
            'negative_text': negative_sample['texts'],
            'label': anchor_sample['labels']
        }


class CLIPDataCollator:
    """Custom collator for CLIP training batches"""
    
    def __init__(self, dataset_type: str = "standard"):
        """
        Initialize collator
        
        Args:
            dataset_type: Type of dataset ('standard', 'hard_negative', 'triplet')
        """
        self.dataset_type = dataset_type
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch of samples"""
        if self.dataset_type == "standard":
            return self._collate_standard(batch)
        elif self.dataset_type == "hard_negative":
            return self._collate_hard_negative(batch)
        elif self.dataset_type == "triplet":
            return self._collate_triplet(batch)
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
    
    def _collate_standard(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate standard image-caption pairs"""
        images = torch.stack([item['images'] for item in batch])
        texts = torch.stack([item['texts'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        return {
            'images': images,
            'texts': texts,
            'labels': labels
        }
    
    def _collate_hard_negative(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate hard negative samples"""
        positive_images = torch.stack([item['positive_image'] for item in batch])
        positive_texts = torch.stack([item['positive_text'] for item in batch])
        negative_images = torch.cat([item['negative_images'] for item in batch], dim=0)
        negative_texts = torch.cat([item['negative_texts'] for item in batch], dim=0)
        labels = torch.stack([item['label'] for item in batch])
        
        return {
            'positive_images': positive_images,
            'positive_texts': positive_texts,
            'negative_images': negative_images,
            'negative_texts': negative_texts,
            'labels': labels
        }
    
    def _collate_triplet(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate triplet samples"""
        anchor_images = torch.stack([item['anchor_image'] for item in batch])
        anchor_texts = torch.stack([item['anchor_text'] for item in batch])
        positive_images = torch.stack([item['positive_image'] for item in batch])
        positive_texts = torch.stack([item['positive_text'] for item in batch])
        negative_images = torch.stack([item['negative_image'] for item in batch])
        negative_texts = torch.stack([item['negative_text'] for item in batch])
        labels = torch.stack([item['label'] for item in batch])
        
        return {
            'anchor_images': anchor_images,
            'anchor_texts': anchor_texts,
            'positive_images': positive_images,
            'positive_texts': positive_texts,
            'negative_images': negative_images,
            'negative_texts': negative_texts,
            'labels': labels
        }


def create_dataloader(dataset: Dataset, batch_size: int, 
                     dataset_type: str = "standard", 
                     shuffle: bool = True, num_workers: int = 4) -> DataLoader:
    """Create dataloader with appropriate collator"""
    collator = CLIPDataCollator(dataset_type)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collator
    )


def main():
    """Example usage of dataset classes"""
    print("CLIP Fine-tuning Datasets")
    print("=" * 40)
    
    print("Available dataset types:")
    print("1. CLIPFineTuneDataset - Standard image-caption pairs")
    print("2. CLIPHardNegativeDataset - With hard negative mining")
    print("3. CLIPTripletDataset - Triplet-based training")
    
    print("\nFeatures:")
    print("- Automatic hard negative mining")
    print("- Triplet generation strategies")
    print("- Custom data collators")
    print("- Support for various data formats")
    print("- Built-in CLIP preprocessing")

if __name__ == "__main__":
    main()

