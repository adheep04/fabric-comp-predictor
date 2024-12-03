import os
from dataset import FabricDataset
from config import get_config  # Ensure your config module is correctly set up
import pickle

def main():

    # Load the configuration
    config = get_config()  # Adjust if your config needs additional parameters

    # Set the dataset path
    data_path = "AIProjectFabricComp//data//raw_data//fabrics"
    
    # Initialize the dataset
    dataset = FabricDataset(data_path=data_path, config=config)
    
    print(dataset.base_fabric_types)

if __name__ == "__main__":
    main()