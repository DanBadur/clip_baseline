#!/usr/bin/env python3
"""
CLIP Fine-tuning Trainer for Improved Top-K Image Retrieval
Implements contrastive learning with hard negative mining and various loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import clip
from tqdm import tqdm
import logging
from datetime import datetime

from fine_tune_utils import (
    CLIPDataAugmentation, 
    CLIPModelFreezer, 
    CLIPTrainingUtilities,
    CLIPDataUtilities
)

class CLIPFineTuneTrainer:
    """Main trainer class for fine-tuning CLIP models"""
    
    def __init__(self, config_path: str):
        """
        Initialize the trainer
        
        Args:
            config_path: Path to training configuration file
        """
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model and components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.training_history = []
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self._initialize_model()
        self._initialize_training_components()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load training configuration"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Set default values
        defaults = {
            'model_name': 'ViT-B/32',
            'learning_rate': 1e-5,
            'batch_size': 32,
            'epochs': 10,
            'temperature': 0.07,
            'margin': 0.5,
            'hard_negative_mining': True,
            'loss_type': 'contrastive',
            'freeze_image_encoder': False,
            'freeze_text_encoder': False,
            'use_augmentation': True,
            'save_checkpoints': True,
            'checkpoint_dir': 'checkpoints',
            'eval_every': 1,
            'patience': 5
        }
        
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
        
        return config
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"training_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _initialize_model(self):
        """Initialize CLIP model"""
        self.logger.info(f"Loading CLIP model: {self.config['model_name']}")
        self.model, self.preprocess = clip.load(
            self.config['model_name'], 
            device=self.device
        )
        
        # Initialize model freezer
        self.model_freezer = CLIPModelFreezer(self.model)
        
        # Apply freezing strategy
        if self.config['freeze_image_encoder']:
            self.model_freezer.freeze_image_encoder(True)
            self.logger.info("Image encoder frozen")
        
        if self.config['freeze_text_encoder']:
            self.model_freezer.freeze_text_encoder(True)
            self.logger.info("Text encoder frozen")
        
        # Print parameter status
        self.model_freezer.print_parameter_status()
        
        # Move to device
        self.model = self.model.to(self.device)
    
    def _initialize_training_components(self):
        """Initialize optimizer, scheduler, and loss function"""
        # Create optimizer
        self.optimizer = CLIPTrainingUtilities.create_optimizer(
            self.model, self.config
        )
        
        # Create scheduler
        lr_schedule = CLIPTrainingUtilities.create_learning_rate_schedule(self.config)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_schedule
        )
        
        # Create loss function
        self.criterion = self._create_loss_function()
        
        self.logger.info(f"Initialized {type(self.optimizer).__name__} optimizer")
        self.logger.info(f"Initialized {type(self.criterion).__name__} loss function")
    
    def _create_loss_function(self) -> nn.Module:
        """Create the appropriate loss function"""
        if self.config['loss_type'] == 'contrastive':
            return ContrastiveLossWithHardNegatives(
                temperature=self.config['temperature'],
                margin=self.config['margin'],
                use_hard_negative_mining=self.config['hard_negative_mining']
            )
        elif self.config['loss_type'] == 'triplet':
            return TripletLoss(margin=self.config['margin'])
        elif self.config['loss_type'] == 'info_nce':
            return InfoNCELoss(temperature=self.config['temperature'])
        else:
            raise ValueError(f"Unknown loss type: {self.config['loss_type']}")
    
    def train(self, train_dataset: Dataset, val_dataset: Optional[Dataset] = None):
        """Main training loop"""
        self.logger.info("Starting training...")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        
        # Training loop
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            self.current_epoch = epoch
            
            # Training phase
            train_loss, train_metrics = self._train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_metrics = None, None
            if val_loader and epoch % self.config['eval_every'] == 0:
                val_loss, val_metrics = self._validate_epoch(val_loader)
            
            # Logging
            self._log_epoch_results(epoch, train_loss, train_metrics, val_loss, val_metrics)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Checkpointing
            if self.config['save_checkpoints']:
                self._save_checkpoint(epoch, val_metrics)
            
            # Early stopping
            if val_metrics and 'retrieval_map' in val_metrics:
                current_metric = val_metrics['retrieval_map']
                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    patience_counter = 0
                    self._save_best_model()
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.config['patience']:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        self.logger.info("Training completed!")
        return self.training_history
    
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        batch_metrics = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            images, texts, labels = self._prepare_batch(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Get features
            image_features = self.model.encode_image(images)
            text_features = self.model.encode_text(texts)
            
            # Normalize features
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            
            # Compute loss
            loss = self.criterion(image_features, text_features, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            batch_metrics.append({
                'loss': loss.item(),
                'batch_size': images.size(0)
            })
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/(batch_idx+1):.4f}"
            })
        
        avg_loss = total_loss / len(train_loader)
        epoch_metrics = self._compute_epoch_metrics(batch_metrics)
        
        return avg_loss, epoch_metrics
    
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        all_image_features = []
        all_text_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                images, texts, labels = self._prepare_batch(batch)
                
                # Get features
                image_features = self.model.encode_image(images)
                text_features = self.model.encode_text(texts)
                
                # Normalize features
                image_features = F.normalize(image_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)
                
                # Compute loss
                loss = self.criterion(image_features, text_features, labels)
                total_loss += loss.item()
                
                # Store features for retrieval evaluation
                all_image_features.append(image_features.cpu())
                all_text_features.append(text_features.cpu())
                all_labels.append(labels.cpu())
        
        # Concatenate all features
        all_image_features = torch.cat(all_image_features, dim=0)
        all_text_features = torch.cat(all_text_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Compute retrieval metrics
        retrieval_metrics = self._compute_retrieval_metrics(
            all_image_features, all_text_features, all_labels
        )
        
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss, retrieval_metrics
    
    def _prepare_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare batch for training"""
        if isinstance(batch, (list, tuple)):
            images, texts, labels = batch
        else:
            images = batch['images']
            texts = batch['texts']
            labels = batch['labels']
        
        return (
            images.to(self.device),
            texts.to(self.device),
            labels.to(self.device)
        )
    
    def _compute_epoch_metrics(self, batch_metrics: List[Dict]) -> Dict:
        """Compute metrics for the epoch"""
        total_loss = sum(m['loss'] for m in batch_metrics)
        total_samples = sum(m['batch_size'] for m in batch_metrics)
        
        return {
            'avg_loss': total_loss / len(batch_metrics),
            'total_samples': total_samples
        }
    
    def _compute_retrieval_metrics(self, image_features: torch.Tensor, 
                                 text_features: torch.Tensor, 
                                 labels: torch.Tensor) -> Dict:
        """Compute retrieval metrics (mAP, Recall@K, etc.)"""
        # Compute similarity matrix
        similarity_matrix = torch.mm(image_features, text_features.T)
        
        # Get top-k predictions
        k_values = [1, 5, 10]
        metrics = {}
        
        for k in k_values:
            # Text-to-image retrieval
            top_k_indices = torch.topk(similarity_matrix, k=k, dim=1)[1]
            text_to_image_recall = self._compute_recall_at_k(
                top_k_indices, labels, k
            )
            
            # Image-to-text retrieval
            top_k_indices = torch.topk(similarity_matrix, k=k, dim=0)[1]
            image_to_text_recall = self._compute_recall_at_k(
                top_k_indices.T, labels, k
            )
            
            metrics[f'recall@{k}_text_to_image'] = text_to_image_recall
            metrics[f'recall@{k}_image_to_text'] = image_to_text_recall
        
        # Compute mAP
        metrics['retrieval_map'] = self._compute_map(similarity_matrix, labels)
        
        return metrics
    
    def _compute_recall_at_k(self, top_k_indices: torch.Tensor, 
                           labels: torch.Tensor, k: int) -> float:
        """Compute recall@k"""
        correct = 0
        total = 0
        
        for i in range(top_k_indices.size(0)):
            if labels[i] < top_k_indices.size(1):  # Valid label
                if labels[i] in top_k_indices[i][:k]:
                    correct += 1
                total += 1
        
        return correct / total if total > 0 else 0.0
    
    def _compute_map(self, similarity_matrix: torch.Tensor, 
                    labels: torch.Tensor) -> float:
        """Compute mean Average Precision"""
        # Sort similarities in descending order
        sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)
        
        map_scores = []
        for i in range(similarity_matrix.size(0)):
            if labels[i] < similarity_matrix.size(1):  # Valid label
                # Find position of correct label
                correct_pos = torch.where(sorted_indices[i] == labels[i])[0]
                if len(correct_pos) > 0:
                    rank = correct_pos[0].item() + 1
                    precision = 1.0 / rank
                    map_scores.append(precision)
        
        return np.mean(map_scores) if map_scores else 0.0
    
    def _log_epoch_results(self, epoch: int, train_loss: float, 
                          train_metrics: Dict, val_loss: Optional[float], 
                          val_metrics: Optional[Dict]):
        """Log results for the epoch"""
        log_msg = f"Epoch {epoch}: Train Loss: {train_loss:.4f}"
        
        if val_loss is not None:
            log_msg += f", Val Loss: {val_loss:.4f}"
        
        if val_metrics:
            log_msg += f", Val mAP: {val_metrics.get('retrieval_map', 0):.4f}"
        
        self.logger.info(log_msg)
        
        # Store in history
        epoch_result = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_metrics': train_metrics,
            'val_loss': val_loss,
            'val_metrics': val_metrics
        }
        self.training_history.append(epoch_result)
    
    def _save_checkpoint(self, epoch: int, val_metrics: Optional[Dict]):
        """Save training checkpoint"""
        checkpoint_dir = Path(self.config['checkpoint_dir'])
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'training_history': self.training_history,
            'best_metric': self.best_metric
        }
        
        if val_metrics:
            checkpoint['val_metrics'] = val_metrics
        
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def _save_best_model(self):
        """Save the best model"""
        best_model_path = Path(self.config['checkpoint_dir']) / "best_model.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'best_metric': self.best_metric
        }, best_model_path)
        
        self.logger.info(f"Saved best model with metric: {self.best_metric:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.training_history = checkpoint.get('training_history', [])
        self.best_metric = checkpoint.get('best_metric', 0.0)
        
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def save_final_model(self, output_path: str):
        """Save the final fine-tuned model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'training_history': self.training_history,
            'best_metric': self.best_metric
        }, output_path)
        
        self.logger.info(f"Saved final model to: {output_path}")


class ContrastiveLossWithHardNegatives(nn.Module):
    """Contrastive loss with hard negative mining for better retrieval"""
    
    def __init__(self, temperature: float = 0.07, margin: float = 0.5, 
                 use_hard_negative_mining: bool = True):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.use_hard_negative_mining = use_hard_negative_mining
    
    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor, 
               labels: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss with hard negative mining"""
        batch_size = image_features.size(0)
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(image_features, text_features.T) / self.temperature
        
        # Create positive mask (diagonal)
        positive_mask = torch.eye(batch_size, device=image_features.device)
        
        # Positive pairs loss
        positive_loss = -torch.log(
            torch.exp(similarity_matrix) + 1e-8
        ) * positive_mask
        positive_loss = positive_loss.sum() / positive_mask.sum()
        
        if self.use_hard_negative_mining:
            # Hard negative mining
            negative_mask = 1 - positive_mask
            
            # Find hardest negatives (highest similarity among negatives)
            hard_negatives = similarity_matrix * negative_mask
            hard_negative_values, _ = torch.topk(
                hard_negatives, 
                k=min(3, batch_size-1),  # Top 3 hardest negatives
                dim=1
            )
            
            # Apply margin-based loss to hard negatives
            hard_negative_loss = torch.clamp(
                hard_negative_values - self.margin, 
                min=0
            ).mean()
            
            return positive_loss + hard_negative_loss
        else:
            # Standard contrastive loss
            negative_loss = torch.log(
                torch.exp(similarity_matrix) + 1e-8
            ) * negative_mask
            negative_loss = negative_loss.sum() / negative_mask.sum()
            
            return positive_loss + negative_loss


class TripletLoss(nn.Module):
    """Triplet loss for learning better feature representations"""
    
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin
    
    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor, 
               labels: torch.Tensor) -> torch.Tensor:
        """Compute triplet loss"""
        batch_size = image_features.size(0)
        
        # Compute pairwise distances
        image_distances = torch.cdist(image_features, image_features)
        text_distances = torch.cdist(text_features, text_features)
        
        # Create positive and negative masks
        positive_mask = torch.eye(batch_size, device=image_features.device)
        negative_mask = 1 - positive_mask
        
        # Find hardest positive and negative pairs
        triplet_losses = []
        
        for i in range(batch_size):
            # Hardest positive (furthest positive)
            positive_distances = image_distances[i] * positive_mask[i]
            hardest_positive = positive_distances.max()
            
            # Hardest negative (closest negative)
            negative_distances = image_distances[i] * negative_mask[i]
            hardest_negative = negative_distances.min()
            
            # Triplet loss
            triplet_loss = torch.clamp(
                hardest_positive - hardest_negative + self.margin, 
                min=0
            )
            triplet_losses.append(triplet_loss)
        
        return torch.stack(triplet_losses).mean()


class InfoNCELoss(nn.Module):
    """InfoNCE loss with temperature scaling"""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor, 
               labels: torch.Tensor) -> torch.Tensor:
        """Compute InfoNCE loss"""
        batch_size = image_features.size(0)
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(image_features, text_features.T) / self.temperature
        
        # Create positive mask (diagonal)
        positive_mask = torch.eye(batch_size, device=image_features.device)
        
        # InfoNCE loss
        exp_sim = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # Average log probability of positive pairs
        loss = -(log_prob * positive_mask).sum(dim=1).mean()
        
        return loss


def main():
    """Example usage of the fine-tuning trainer"""
    print("CLIP Fine-tuning Trainer")
    print("=" * 40)
    
    # Example configuration
    config = {
        'model_name': 'ViT-B/32',
        'learning_rate': 1e-5,
        'batch_size': 32,
        'epochs': 10,
        'temperature': 0.07,
        'margin': 0.5,
        'hard_negative_mining': True,
        'loss_type': 'contrastive',
        'freeze_image_encoder': False,
        'freeze_text_encoder': False,
        'use_augmentation': True,
        'save_checkpoints': True,
        'checkpoint_dir': 'checkpoints',
        'eval_every': 1,
        'patience': 5
    }
    
    print("Available loss functions:")
    print("1. ContrastiveLossWithHardNegatives - Enhanced contrastive learning")
    print("2. TripletLoss - Triplet-based learning")
    print("3. InfoNCELoss - InfoNCE with temperature scaling")
    
    print("\nTraining features:")
    print("- Hard negative mining for better retrieval")
    print("- Multiple loss function options")
    print("- Comprehensive evaluation metrics")
    print("- Checkpointing and early stopping")
    print("- Learning rate scheduling")

if __name__ == "__main__":
    main()

