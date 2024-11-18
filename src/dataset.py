import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as T

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        """
        Args:
            annotations_file (str): Path to the CSV file with annotations.
            img_dir (str): Directory with all the images.
            transform (callable, optional): Transform to be applied to images.
        """
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform or T.Compose([
            T.Resize((224, 224)),  # Resize to ResNet's expected size
            T.ToTensor(),          # Convert to tensor
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for ResNet
        ])
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to fetch.
        Returns:
            tuple: (image tensor, label tensor)
        """
        # Load the image
        img_name = self.img_labels.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        image = read_image(img_path).float() / 255.0  # Scale image to [0, 1]

        # Load the label
        label_str = self.img_labels.iloc[idx, 1]  # e.g., "cotton:70,polyester:30"
        label_dict = {k: float(v) for k, v in (item.split(':') for item in label_str.split(','))}
        
        # Convert label to tensor with percentage values
        label_tensor = torch.tensor([value / 100.0 for value in label_dict.values()])  # Directly create tensor
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label_tensor
