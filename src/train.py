import torch.nn as nn
import torch.optim
import torch.utils.data

from config import get_config
from dataset import FabricDataset, get_dataloaders
from model import get_model

import numpy as np
import random
from tqdm import tqdm  
import logging


def train_model(config=get_config()):
    device = torch.device('cuda')
    print(f"Using device {device}")
    
    train_dataloader, test_dataloader = get_dataloaders(config)
    model = get_model(config).to(device)
    
    # Initialize weights using xavier transform
    for name, p in model.named_parameters():
        if p.dim() > 1 and "backbone" not in name:  # Skip pretrained layers
            nn.init.xavier_uniform_(p)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)


    
    