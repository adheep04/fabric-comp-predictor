import torch.nn as nn
import torch.optim
import torch.utils.data

from config import get_config
from dataset import FabricDataset, get_dataloaders
from model import get_model
from torch.nn.utils import clip_grad_norm_

import numpy as np
import random
from tqdm import tqdm  
import logging
import os

from torch.optim import lr_scheduler

from model import FeatureExtractor, SingleHeadAttention, ClothingEmbedding, FusionLayer, FeedForwardPrediction

def train_model(config=get_config()):
    device = torch.device('cuda')
    print(f"Using device {device}")
    
    train_dataloader, val_dataloader = get_dataloaders(config)
    model = get_model(config).to(device)
    
    forward_outputs = {}
    gradient_norms = {}

    # Forward Hook
    def forward_hook(module, input, output):
        module_name = module.__class__.__name__
        forward_outputs[module_name] = output.detach()

    # Backward Hook
    def backward_hook(module, grad_input, grad_output):
        module_name = module.__class__.__name__
        gradient_norms[module_name] = grad_output[0].norm().item()  # Norm of the first gradient output

    hooks = []
    for module_name, module in model.named_modules():
        if isinstance(module, (FeatureExtractor, SingleHeadAttention, ClothingEmbedding, FusionLayer, FeedForwardPrediction)):
            hooks.append(module.register_forward_hook(forward_hook))
            hooks.append(module.register_backward_hook(backward_hook))
    
    # Initialize weights
    initialize_weights(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    
    # Initialize ReduceLROnPlateau scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.8, verbose=True)
    
    # Binary Cross-Entropy Loss
    loss_fn = BinaryAndRegressionLoss(threshold=0, alpha=config['alpha'], beta=config['beta'], sparsity_penalty=config['sparsity_penalty'], c_penalty=config['c_penalty'])

    # Precompute fabric weights (based on how often a sample contains that fabric)
    fabric_counts_dict = config['fabric_counts']  # Extract the dictionary
    fabric_counts = list(fabric_counts_dict.values())  # Convert values to a list
    total_count = sum(fabric_counts)
    
    class_weights = [total_count / c for c in fabric_counts]  # Inverse frequency
    total_weight = sum(class_weights)
    class_weights = [w * np.log(w*w) / np.sqrt(total_weight*np.abs(w - w*w)) for w in class_weights]  # Normalize so sum is 1
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Index of "unknown"
    unknown_index = config['fabric_types'].index('unknown')  # Assuming fabric_types lists fabrics in order

    best_val_loss = float("inf")
    checkpoint_dir = config.get("checkpoint_dir", "./checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(config['num_epochs']):
        print(f"Starting Epoch {epoch + 1}/{config['num_epochs']}")

        # Training Phase
        model.train()
        running_loss = 0.0
        train_iterator = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")

        for step, batch in enumerate(train_iterator):
            
            # get batch from dataloader
            images, clothing_vectors, true_fabrics = batch
            images, clothing_vectors, true_fabrics = images.to(device), clothing_vectors.to(device), true_fabrics.to(device)

            # Remove "unknown" from loss calculation
            true_fabrics_no_unknown = true_fabrics.clone()
            true_fabrics_no_unknown[:, unknown_index] = 0.0

            # get predictions from mode
            predicted_fabrics = model(images, clothing_vectors)

            # calculate loss
            total_loss, bce_loss, mse_loss, sparsity_penalty, c_penalty = loss_fn(predicted_fabrics, true_fabrics * 10, class_weights_tensor)

            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=config['max_norm'])  # Clip gradients with norm > 1.0
            optimizer.step()

            # Update running loss
            running_loss += total_loss.item()
            train_iterator.set_postfix(loss=total_loss.item())
            
            if (epoch == 0 and step == 0) or (step + 1) % 25 == 0:  # Log periodically
                print(f"Step {step + 1} - Monitoring Bottlenecks...")
                
                # Forward Monitoring
                for module_name, output in forward_outputs.items():
                    print(f"{module_name} - Output Mean: {output.mean().item():.4f}, Variance: {output.var().item():.4f}")

                # Backward Monitoring
                for module_name, grad_norm in gradient_norms.items():
                    print(f"{module_name} - Gradient Norm: {grad_norm:.4f}")
            
            if (epoch == 0 and step == 0) or (step + 1) % 25 == 0:  # Log periodically
                for i in range(4):
                    print()
                    print(f"pred: {(torch.topk(predicted_fabrics[i], 4)[1]).cpu().numpy().tolist()}")
                    print(f"true: {(torch.topk(true_fabrics_no_unknown[i], 4)[1]).cpu().numpy().tolist()}")
                    print()
                print(f"BCE Loss: {torch.mean(bce_loss).item()}")
                print(f"MSE Loss: {torch.mean(mse_loss).item()}")
                print(f"Sparsity Penalty: {sparsity_penalty.item()}")
                print(f"Count Penalty: {c_penalty.item()}")
            

        train_loss = running_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} Training Loss: {train_loss:.4f}")

        # Validation Phase
        model.eval()
        val_running_loss = 0.0
                    
        with torch.no_grad():
            val_iterator = tqdm(val_dataloader, desc=f"Validating Epoch {epoch + 1}")

            for step, batch in enumerate(val_iterator):
                images, clothing_vectors, true_fabrics = batch
                images, clothing_vectors, true_fabrics = images.to(device), clothing_vectors.to(device), true_fabrics.to(device)

                predicted_fabrics = model(images, clothing_vectors)
                
                total_loss, bce_loss, mse_loss, sparsity_penalty, c_penalty = loss_fn(predicted_fabrics, true_fabrics, class_weights_tensor)

                # Update validation loss
                val_running_loss += total_loss.item()
                
                if (step + 1) % 50 == 0:  # Log periodically
                    print(f'___________ VALIDATION AT STEP {50 * step - 1}______________________________')
                    print(f"Predictions (Sample): {predicted_fabrics[:4]}")
                    print(f"Targets (Sample): {true_fabrics_no_unknown[:4]}")
                    print(f"class-wide accuracy: {calculate_class_wise_accuracy(predicted_fabrics, true_fabrics_no_unknown)}")

            

        val_loss = val_running_loss / len(val_dataloader)
        print(f"Epoch {epoch + 1} Validation Loss: {val_loss:.4f}")

        # Adjust the learning rate
        scheduler.step(val_loss)  
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(checkpoint_dir, f"best_model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved at {checkpoint_path}")
        
    print("Training completed.")


def calculate_class_wise_accuracy(predictions, targets, tolerance=0.05):
    """
    Calculate class-wise accuracy for regression predictions.

    Args:
        predictions (torch.Tensor): Predicted fabric compositions (batch_size, num_classes).
        targets (torch.Tensor): Ground truth fabric compositions (batch_size, num_classes).
        tolerance (float): Allowed difference between prediction and target for a correct result.

    Returns:
        dict: A dictionary with class indices as keys and accuracies as values.
    """
    within_tolerance = torch.abs(predictions - targets) <= tolerance  # Boolean tensor
    class_accuracies = within_tolerance.float().mean(dim=0)  # Mean across batch for each class
    return class_accuracies
    
import torch
import torch.nn as nn

class BinaryAndRegressionLoss(nn.Module):
    def __init__(self, threshold=0.1, alpha=2.4, beta=0.7, sparsity_penalty=0.08, c_penalty=2.0):
        super(BinaryAndRegressionLoss, self).__init__()
        
        # Threshold to turn continuous targets and predictions into binary
        self.threshold = threshold

        # penalties
        self.c_penalty = c_penalty
        self.sparsity_penalty_weight = sparsity_penalty
        
        # Weight for the classification loss and regression loss
        self.alpha = alpha  # Weight for the binary classification loss
        self.beta = beta    # Weight for the regression (MSE) loss
        
        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')  # BCE loss (with logits)
        self.mse_loss = nn.MSELoss(reduction='none')  # MSE Loss (regression)

    def forward(self, predictions, targets, fabric_weights=None):
        
        """
        Custom combined loss function that combines binary classification BCE loss and regression MSE loss.
        
        :param predictions: Tensor of predicted fabric proportions (batch_size, num_fabrics).
        :param targets: Tensor of target fabric proportions (batch_size, num_fabrics).
        :param fabric_weights: Tensor of fabric weights (batch_size, num_fabrics), optional, for weighted BCE loss.
        """
        
        binary_predictions = (predictions > 0).float()
        binary_targets = (targets > 0).float()

        bce_loss = self.bce_loss(binary_predictions, binary_targets)
        
        
        # If fabric_weights are provided, apply them to the BCE loss
        if fabric_weights is not None:
            bce_loss_weighted = (bce_loss * fabric_weights)  # Weight BCE loss based on fabric distribution
            weighted_bce_loss = torch.mean(bce_loss_weighted)  # Mean weighted BCE loss
        else:
            weighted_bce_loss = torch.mean(bce_loss)  # Default to unweighted BCE loss

        mse_loss = self.mse_loss(predictions, targets)
        weighted_mse_loss = torch.mean(mse_loss)
        
        predicted_count = (predictions > 0.3).sum()
        target_count = (targets > 0).sum()
        c_penalty = abs(predicted_count - target_count) * self.c_penalty

        sparsity_penalty = torch.mean((binary_targets == 0) * predictions.abs())
        
        mse_portion = self.beta * weighted_mse_loss
        bce_portion = self.alpha * weighted_bce_loss
        sparsity_portion = self.sparsity_penalty_weight * sparsity_penalty

        total_loss = bce_portion + mse_portion + sparsity_portion + c_penalty
        
        return total_loss, bce_portion, mse_portion, sparsity_penalty, c_penalty

# Initialize weights for the network
def initialize_weights(model):
    for name, p in model.named_parameters():
        if p.dim() > 1:  # Only initialize weights for layers with weights (not biases)
            if 'backbone' in name:  # Skip backbone layers if you want to freeze them
                continue
            elif 'conv' in name or 'linear' in name:  # Check if it's a convolutional or linear layer
                if isinstance(p, nn.Conv2d) or isinstance(p, nn.Linear):
                    # Apply Kaiming Normal for ReLU layers (most common in ResNet)
                    nn.init.kaiming_normal_(p, mode='fan_out', nonlinearity='relu')
            else:
                # For non-ReLU layers, use Xavier Normal or Orthogonal Initialization
                nn.init.xavier_normal_(p)
                
        # Optional: Initialize biases to 0 if not already initialized
        if 'bias' in name:
            nn.init.zeros_(p)