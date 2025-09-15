#!/usr/bin/env python3
"""
Utility functions for CLIP fine-tuning
Includes data augmentation, model freezing strategies, and training utilities
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
import json
from pathlib import Path

class CLIPDataAugmentation:
    """Data augmentation utilities for CLIP fine-tuning"""
    
    def __init__(self, config: Dict):
        """
        Initialize data augmentation
        
        Args:
            config: Augmentation configuration dictionary
        """
        self.config = config
        self.transforms = self._create_transforms()
    
    def _create_transforms(self) -> transforms.Compose:
        """Create augmentation transforms"""
        transform_list = []
        
        if self.config.get('use_augmentation', True):
            # Color jittering
            if self.config.get('color_jitter', 0) > 0:
                transform_list.append(
                    transforms.ColorJitter(
                        brightness=self.config['color_jitter'],
                        contrast=self.config['color_jitter'],
                        saturation=self.config['color_jitter'],
                        hue=self.config['color_jitter'] * 0.5
                    )
                )
            
            # Random horizontal flip
            if self.config.get('random_horizontal_flip', 0) > 0:
                transform_list.append(
                    transforms.RandomHorizontalFlip(p=self.config['random_horizontal_flip'])
                )
            
            # Random rotation
            if self.config.get('random_rotation', 0) > 0:
                transform_list.append(
                    transforms.RandomRotation(degrees=self.config['random_rotation'])
                )
            
            # Random crop and resize
            if self.config.get('random_crop', 1.0) < 1.0:
                crop_size = int(224 * self.config['random_crop'])
                transform_list.append(
                    transforms.RandomCrop(crop_size)
                )
                transform_list.append(
                    transforms.Resize((224, 224))
                )
            
            # Random erasing (applied after normalization)
            if self.config.get('random_erasing', 0) > 0:
                transform_list.append(
                    transforms.RandomErasing(p=self.config['random_erasing'])
                )
        
        return transforms.Compose(transform_list)
    
    def apply_augmentation(self, image: Image.Image) -> Image.Image:
        """Apply augmentation to a single image"""
        return self.transforms(image)
    
    def apply_batch_augmentation(self, images: List[Image.Image]) -> List[Image.Image]:
        """Apply augmentation to a batch of images"""
        return [self.apply_augmentation(img) for img in images]

class CLIPModelFreezer:
    """Utilities for freezing/unfreezing CLIP model components"""
    
    def __init__(self, model: nn.Module):
        """
        Initialize model freezer
        
        Args:
            model: CLIP model to manage
        """
        self.model = model
        self.original_state = {}
        self._save_original_state()
    
    def _save_original_state(self):
        """Save original model state for restoration"""
        for name, param in self.model.named_parameters():
            self.original_state[name] = param.requires_grad
    
    def freeze_component(self, component_name: str, freeze: bool = True):
        """
        Freeze or unfreeze a specific model component
        
        Args:
            component_name: Name of component to freeze/unfreeze
            freeze: Whether to freeze (True) or unfreeze (False)
        """
        for name, param in self.model.named_parameters():
            if component_name in name:
                param.requires_grad = not freeze
                print(f"{'Frozen' if freeze else 'Unfrozen'}: {name}")
    
    def freeze_image_encoder(self, freeze: bool = True):
        """Freeze/unfreeze image encoder"""
        self.freeze_component("visual", freeze)
    
    def freeze_text_encoder(self, freeze: bool = True):
        """Freeze/unfreeze text encoder"""
        self.freeze_component("transformer", freeze)
    
    def freeze_projections(self, freeze: bool = True):
        """Freeze/unfreeze projection layers"""
        self.freeze_component("visual.proj", freeze)
        self.freeze_component("text_projection", freeze)
    
    def freeze_all_except_classifier(self, freeze: bool = True):
        """Freeze all layers except the final classifier"""
        for name, param in self.model.named_parameters():
            if "classifier" not in name and "fc" not in name:
                param.requires_grad = not freeze
    
    def restore_original_state(self):
        """Restore original parameter states"""
        for name, param in self.model.named_parameters():
            if name in self.original_state:
                param.requires_grad = self.original_state[name]
        print("Restored original model state")
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get list of trainable parameters"""
        return [param for param in self.model.parameters() if param.requires_grad]
    
    def count_trainable_parameters(self) -> int:
        """Count number of trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def print_parameter_status(self):
        """Print status of all parameters"""
        print("Model Parameter Status:")
        print("-" * 50)
        
        total_params = 0
        trainable_params = 0
        
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                status = "TRAINABLE"
            else:
                status = "FROZEN"
            
            print(f"{name}: {status} ({param.numel():,} params)")
        
        print("-" * 50)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {total_params - trainable_params:,}")

class CLIPTrainingUtilities:
    """Utility functions for CLIP training"""
    
    @staticmethod
    def create_learning_rate_schedule(config: Dict) -> Callable:
        """
        Create learning rate schedule function
        
        Args:
            config: Training configuration
            
        Returns:
            Function that takes epoch and returns learning rate
        """
        scheduler_type = config.get('scheduler', 'cosine')
        
        if scheduler_type == 'cosine':
            def cosine_schedule(epoch):
                initial_lr = config['learning_rate']
                min_lr = config.get('min_lr', initial_lr * 0.01)
                max_epochs = config['epochs']
                
                # Cosine annealing with warm restarts
                if config.get('warm_restarts', False):
                    restart_epoch = config.get('restart_epoch', max_epochs)
                    epoch_in_cycle = epoch % restart_epoch
                    return min_lr + (initial_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * epoch_in_cycle / restart_epoch))
                else:
                    return min_lr + (initial_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * epoch / max_epochs))
            
            return cosine_schedule
        
        elif scheduler_type == 'step':
            def step_schedule(epoch):
                initial_lr = config['learning_rate']
                step_size = config.get('step_size', 30)
                gamma = config.get('gamma', 0.1)
                
                return initial_lr * (gamma ** (epoch // step_size))
            
            return step_schedule
        
        elif scheduler_type == 'exponential':
            def exp_schedule(epoch):
                initial_lr = config['learning_rate']
                decay_rate = config.get('decay_rate', 0.95)
                
                return initial_lr * (decay_rate ** epoch)
            
            return exp_schedule
        
        else:
            # Constant learning rate
            def constant_schedule(epoch):
                return config['learning_rate']
            
            return constant_schedule
    
    @staticmethod
    def create_optimizer(model: nn.Module, config: Dict) -> torch.optim.Optimizer:
        """
        Create optimizer with different parameter groups
        
        Args:
            model: Model to optimize
            config: Training configuration
            
        Returns:
            Optimizer instance
        """
        # Separate parameters by type if specified
        if config.get('separate_parameter_groups', False):
            param_groups = CLIPTrainingUtilities._create_parameter_groups(model, config)
        else:
            param_groups = [{'params': model.parameters()}]
        
        # Add common optimizer parameters
        for group in param_groups:
            group['lr'] = config['learning_rate']
            if 'weight_decay' in config:
                group['weight_decay'] = config['weight_decay']
        
        # Create optimizer
        optimizer_name = config.get('optimizer', 'adamw').lower()
        
        if optimizer_name == 'adamw':
            return optim.AdamW(param_groups)
        elif optimizer_name == 'adam':
            return optim.Adam(param_groups)
        elif optimizer_name == 'sgd':
            return optim.SGD(param_groups, momentum=config.get('momentum', 0.9))
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    @staticmethod
    def _create_parameter_groups(model: nn.Module, config: Dict) -> List[Dict]:
        """Create parameter groups with different learning rates"""
        param_groups = []
        
        # Default group
        default_params = []
        # Image encoder group (usually lower learning rate)
        image_params = []
        # Text encoder group
        text_params = []
        # Projection layers group
        proj_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'visual' in name:
                    image_params.append(param)
                elif 'transformer' in name:
                    text_params.append(param)
                elif 'proj' in name or 'text_projection' in name:
                    proj_params.append(param)
                else:
                    default_params.append(param)
        
        # Add groups with different learning rates
        if default_params:
            param_groups.append({'params': default_params})
        
        if image_params:
            image_lr = config.get('image_encoder_lr', config['learning_rate'] * 0.1)
            param_groups.append({'params': image_params, 'lr': image_lr})
        
        if text_params:
            text_lr = config.get('text_encoder_lr', config['learning_rate'] * 0.1)
            param_groups.append({'params': text_params, 'lr': text_lr})
        
        if proj_params:
            proj_lr = config.get('projection_lr', config['learning_rate'] * 2.0)
            param_groups.append({'params': proj_params, 'lr': proj_lr})
        
        return param_groups
    
    @staticmethod
    def create_loss_function(config: Dict) -> nn.Module:
        """
        Create loss function with optional label smoothing
        
        Args:
            config: Training configuration
            
        Returns:
            Loss function
        """
        loss_type = config.get('loss_type', 'cross_entropy')
        
        if loss_type == 'cross_entropy':
            if config.get('label_smoothing', 0) > 0:
                return nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
            else:
                return nn.CrossEntropyLoss()
        
        elif loss_type == 'focal':
            from focal_loss import FocalLoss
            return FocalLoss(alpha=config.get('focal_alpha', 1.0), 
                           gamma=config.get('focal_gamma', 2.0))
        
        elif loss_type == 'contrastive':
            # Custom contrastive loss
            return CLIPTrainingUtilities.ContrastiveLoss(
                temperature=config.get('temperature', 0.07),
                margin=config.get('margin', 0.5)
            )
        
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    class ContrastiveLoss(nn.Module):
        """Custom contrastive loss for CLIP training"""
        
        def __init__(self, temperature: float = 0.07, margin: float = 0.5):
            super().__init__()
            self.temperature = temperature
            self.margin = margin
        
        def forward(self, image_features: torch.Tensor, text_features: torch.Tensor, 
                   labels: torch.Tensor) -> torch.Tensor:
            """
            Compute contrastive loss
            
            Args:
                image_features: Image feature vectors
                text_features: Text feature vectors
                labels: Ground truth labels
                
            Returns:
                Contrastive loss value
            """
            # Normalize features
            image_features = nn.functional.normalize(image_features, dim=-1)
            text_features = nn.functional.normalize(text_features, dim=-1)
            
            # Compute similarity matrix
            similarity_matrix = torch.mm(image_features, text_features.T) / self.temperature
            
            # Create positive mask
            positive_mask = torch.eye(image_features.size(0), device=image_features.device)
            
            # Positive pairs loss
            positive_loss = -torch.log(torch.exp(similarity_matrix) + 1e-8) * positive_mask
            positive_loss = positive_loss.sum() / positive_mask.sum()
            
            # Negative pairs loss (hard negative mining)
            negative_mask = 1 - positive_mask
            hard_negatives = similarity_matrix * negative_mask
            hard_negatives = hard_negatives - self.margin
            hard_negatives = torch.clamp(hard_negatives, min=0)
            
            negative_loss = hard_negatives.sum() / negative_mask.sum()
            
            return positive_loss + negative_loss

class CLIPDataUtilities:
    """Utilities for CLIP data handling"""
    
    @staticmethod
    def create_caption_variations(caption: str, config: Dict) -> List[str]:
        """
        Create variations of a caption for data augmentation
        
        Args:
            caption: Original caption
            config: Augmentation configuration
            
        Returns:
            List of caption variations
        """
        variations = [caption]
        
        if not config.get('augment_captions', False):
            return variations
        
        # Synonym replacement
        if config.get('synonym_replacement', False):
            synonyms = config.get('synonym_dict', {})
            for word, syns in synonyms.items():
                if word.lower() in caption.lower():
                    for syn in syns[:2]:  # Limit to 2 synonyms
                        new_caption = caption.replace(word, syn)
                        variations.append(new_caption)
        
        # Paraphrasing (simple template-based)
        if config.get('paraphrasing', False):
            templates = [
                "This shows {caption}",
                "A picture of {caption}",
                "An image depicting {caption}",
                "This is {caption}"
            ]
            
            for template in templates:
                new_caption = template.format(caption=caption)
                variations.append(new_caption)
        
        return variations[:config.get('max_caption_variations', 5)]
    
    @staticmethod
    def validate_image_caption_pair(image_path: str, caption: str) -> bool:
        """
        Validate image-caption pair
        
        Args:
            image_path: Path to image file
            caption: Caption text
            
        Returns:
            True if pair is valid, False otherwise
        """
        # Check if image exists and is readable
        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception:
            return False
        
        # Check caption length and content
        if not caption or len(caption.strip()) < 3:
            return False
        
        if len(caption) > 1000:  # Reasonable limit
            return False
        
        return True
    
    @staticmethod
    def create_balanced_dataset(image_caption_pairs: List[Dict], 
                              balance_strategy: str = 'category') -> List[Dict]:
        """
        Create balanced dataset to avoid bias
        
        Args:
            image_caption_pairs: List of image-caption pairs
            balance_strategy: Strategy for balancing ('category', 'length', 'random')
            
        Returns:
            Balanced list of pairs
        """
        if balance_strategy == 'random':
            random.shuffle(image_caption_pairs)
            return image_caption_pairs
        
        elif balance_strategy == 'category':
            # Group by category and sample equally
            categories = {}
            for pair in image_caption_pairs:
                category = pair.get('category', 'unknown')
                if category not in categories:
                    categories[category] = []
                categories[category].append(pair)
            
            # Find minimum category size
            min_size = min(len(pairs) for pairs in categories.values())
            
            # Sample equally from each category
            balanced_pairs = []
            for category_pairs in categories.values():
                balanced_pairs.extend(random.sample(category_pairs, min_size))
            
            random.shuffle(balanced_pairs)
            return balanced_pairs
        
        elif balance_strategy == 'length':
            # Group by caption length and sample equally
            length_groups = {}
            for pair in image_caption_pairs:
                caption_length = len(pair['caption'])
                length_bin = caption_length // 20  # 20-character bins
                
                if length_bin not in length_groups:
                    length_groups[length_bin] = []
                length_groups[length_bin].append(pair)
            
            # Find minimum group size
            min_size = min(len(pairs) for pairs in length_groups.values())
            
            # Sample equally from each group
            balanced_pairs = []
            for group_pairs in length_groups.values():
                balanced_pairs.extend(random.sample(group_pairs, min_size))
            
            random.shuffle(balanced_pairs)
            return balanced_pairs
        
        else:
            return image_caption_pairs

def main():
    """Example usage of utility functions"""
    print("CLIP Fine-tuning Utilities")
    print("=" * 40)
    
    # Example configuration
    config = {
        'use_augmentation': True,
        'color_jitter': 0.4,
        'random_horizontal_flip': 0.5,
        'random_rotation': 10,
        'random_crop': 0.9,
        'augment_captions': True,
        'max_caption_variations': 3
    }
    
    print("Available utilities:")
    print("1. CLIPDataAugmentation - Image augmentation")
    print("2. CLIPModelFreezer - Model parameter management")
    print("3. CLIPTrainingUtilities - Training helpers")
    print("4. CLIPDataUtilities - Data handling helpers")
    
    print("\nExample usage:")
    print("augmenter = CLIPDataAugmentation(config)")
    print("freezer = CLIPModelFreezer(model)")
    print("freezer.freeze_image_encoder(True)")

if __name__ == "__main__":
    main()
