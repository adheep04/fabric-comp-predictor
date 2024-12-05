import os
from dataset import FabricDataset
from config import get_config  # Ensure your config module is correctly set up
import pickle

from train import train_model

def main():
    config = get_config()
    train_model(config)
                 
                
    

if __name__ == "__main__":
     main()