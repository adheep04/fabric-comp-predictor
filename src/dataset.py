import os
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split, DataLoader
import torchvision.transforms as T
import pickle
from config import get_config
from PIL import Image
import random
import re

class FabricDataset(Dataset):
    def __init__(self, data_path, config=get_config(), training=True):
        self.data_path = data_path
        self.training = training
        self.inf_transform = T.Compose([
            T.Resize((224, 224)),  # Resize to ResNet's expected size
            T.ToTensor(),          # Convert to tensor
            T.Normalize(mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225])  # Normalize for ResNet
        ])
        
        
        # Define the training transforms
        self.train_transforms = T.Compose([
            # Resize the image to 256x256, then take a random crop of 224x224
            T.Resize((256, 256)),
            T.RandomCrop(224),

            # Apply random horizontal and vertical flips
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),

            # Adjust brightness, contrast, saturation, and hue
            T.ColorJitter(
                brightness=0.15,  # Randomly change brightness
                contrast=0.15,    # Randomly change contrast
                saturation=0.15,  # Randomly change saturation
                hue=0.15         # Randomly change hue
            ),

            # Randomly rotate the image by up to 15 degrees
            T.RandomRotation(degrees=13),

            # Apply GaussianBlur with reduced intensity and probability
            T.RandomApply(
                [T.GaussianBlur(kernel_size=(3, 3), sigma=(0.15, 1.15))],
                p=0.38  # 30% chance of applying GaussianBlur
            ),

            # Convert the image to a tensor
            T.ToTensor(),

            # Normalize using mean and std of ImageNet
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # initialize data
        self.base_fabric_types = []
        self.clothing_types = []
        self.data_info = {}
        
        # gives the number of samples that have a given fabric
        self.fabrics_count = {}
        
        # gives the total fabric composition of the dataset
        self.fabrics_comp = {}
        
        # items per clothing
        self.clothings_count = {}
        
        self.build_data(config)
        self.config = config
        
        '''
        dataset structure: /fabrics/fabric_type/sample_number/[[img1.png, img2.png, ...], tag.txt]
        '''
        # {
        #   1 : {
        #       clothing_type: 'clothing_type'
        #       base_comp: {'fabric_1' : num, 'fabric_2' : num, ...}
        #       folder_label: 'type'
        #       paths: [Path(path1), Path(path2), ...]
        #       unsure: [type]
        #           }   
        #       },
        #   2 : {
        #       clothing_type: 'clothing_type'
        #       base_comp: {'fabric_1' : num, 'fabric_2' : num, ...}
        #       folder_label: 'type'
        #       paths: [Path(path1), Path(path2), ...]
        #       unsure: [type]
        #           }   
        #       }, 
        #   ...
        # }
        
    def __len__(self):
        return len(self.data_info)
    
    def __getitem__(self, index):
        actual_key = list(self.data_info.keys())[index]  # Map index to data_info key
        sample_info = self.data_info[actual_key]

        sample_clothing = sample_info['clothing_type']
        sample_base_comp = sample_info['base_comp']
        sample_img_paths = sample_info['paths']
        
        # only take the first 4 not 5
        sample_img_paths = sample_img_paths[:4]
        
        clothing_type_vector = torch.tensor([1 if clothing == sample_clothing else 0 for clothing in self.clothing_types], dtype=torch.float32)
        fabric_comp_vector = torch.tensor(self.comp_to_vector(sample_base_comp))
        
        # load images
        # image_tensors = (4, 3, 224, 224) = (num_images_per_sample, RGB_channels, height, width)
        image_tensors = torch.stack([self.train_transforms(Image.open(img_path)) if self.training else self.inf_transform(Image.open(img_path)) for img_path in sample_img_paths])
        return (image_tensors, clothing_type_vector, fabric_comp_vector)    
    
    # initializes all 4 data fields
    def build_data(self, config):
        path = config['data_path']
        # iterates through every subfolder in datafolder
        for fabric_type in os.listdir(path):
            # skip unclassified and utility folders
            if fabric_type in ['Unclassified', 'Utilities']:
                continue
            fabric_type_path = os.path.join(path, fabric_type)
            self.parse_fabric_samples(fabric_type_path)
             
    # parses all samples and updates fields given subfolder path
    def parse_fabric_samples(self, type_path):
        def add_if_new(item, list):
            if isinstance(item, str) and item not in list:  # Ensure only valid strings
                list.append(item)
                
        if os.path.basename(type_path).lower() in ['unclassified', 'utilities']:
            print(f"Skipping Unclassified or Utilities folder: {type_path}")
            return
        
        # iterates through every sample folder in subfolder
        for sample_folder in os.listdir(type_path):
            sample_folder_path = os.path.join(type_path, sample_folder)
            
            
            try:
                num_sample, paths, fabric_type = self.validate_sample(sample_folder_path)
            except Exception as e:
                print(f"Skipping invalid sample: {e}")
                continue  # Skip the current sample
            
            
            clothing_type, comp = self.parse_tag(sample_folder_path)
    
            # update self.fabric_comp and self.fabric_count
            for fabric in comp:
                add_if_new(fabric, self.base_fabric_types)
                if fabric in self.fabrics_comp:
                    self.fabrics_comp[fabric] += comp[fabric]
                else:
                    self.fabrics_comp[fabric] = comp[fabric]

                if fabric in self.fabrics_count:
                    self.fabrics_count[fabric] += 1
                else:
                    self.fabrics_count[fabric] = 1
                
            
            # update self.clothing_types
            # print(f'type being added to clothing_types {type}')
            add_if_new(clothing_type, self.clothing_types)

            # update self.clothing_type
            if clothing_type in self.clothings_count:
                self.clothings_count[clothing_type] += 1
            else:
                self.clothings_count[clothing_type] = 1
            
            
            self.data_info[num_sample] = {
                'clothing_type': clothing_type,
                'base_comp': comp,
                'paths': paths,
                'folder_label': fabric_type,
        }

    # Returns [clothing_type, composition{}]
    def parse_tag(self, sample_folder_path):
        def is_unsure(amount):
            return amount.endswith('?')

        def validate_comp(comp, tag_path):
            total = sum(comp.values())
            remaining = 100 - total

            if remaining < 0:
                raise Exception(f'Composition total exceeds 100 at {tag_path}')
            if remaining > 20:
                raise Exception(f'Composition total not at least 80 at {tag_path}')

            comp['unknown'] = max(0, remaining)

        fabric_groups = {
            'saduk silk': 'silk', 'satin silk': 'silk', 'silk': 'silk',
            'naylon': 'nylon', 'polyamide (nylon)': 'nylon',
            'leather': 'leather', 'synthetic leather': 'leather',
            'li': 'linen',
            'pes': 'polyester',
            'ray': 'rayon',
            'elastane': 'spandex', 'sp': 'spandex', 'lycra': 'spandex'
        }

        def normalize_fabric_name(fabric_name):
            fabric_name = re.sub(r'\s+', ' ', fabric_name.strip().lower())
            return fabric_groups.get(fabric_name, fabric_name)

        comp = {}
        tag_path = os.path.join(sample_folder_path, 'tag.txt')

        try:
            with open(tag_path, 'r') as tag:
                # Process first line
                clothing_type = tag.readline().strip().lower()
                if not clothing_type:
                    raise Exception(f'Clothing type malformed in {tag_path}')

                # Process second line
                comp_parts = tag.readline().strip()

                # Handle single fabric with 100% composition
                if not any(char.isdigit() for char in comp_parts):
                    fabric_name = normalize_fabric_name(comp_parts)
                    if fabric_name:
                        comp[fabric_name] = 100
                        return [clothing_type, comp]
                    raise Exception(f'Malformed composition line in {tag_path}: Empty or invalid line')

                # Updated regex matching logic
                matches = re.findall(r'([a-zA-Z\s]+(?:\([a-zA-Z\s]+\))?)\s*(\d+\??)', comp_parts)
                if not matches:
                    raise Exception(f'Malformed composition line in {tag_path}: "{comp_parts}"')

                for fabric, amount_str in matches:
                    fabric = normalize_fabric_name(fabric)
                    if fabric in ['r check', ' ']:
                        continue  # Ignore invalid patterns
                    amount = int(amount_str[:-1]) if is_unsure(amount_str) else int(amount_str)
                    comp[fabric] = comp.get(fabric, 0) + amount

                validate_comp(comp, tag_path)
            
                return (clothing_type, comp)
        except Exception as e:
            raise Exception(f"Error parsing tag at {tag_path}: {str(e)}")
       
    # validates sample folder and returns [sample_number, image_paths[]]
    # doesn't validate tag
    def validate_sample(self, sample_folder):
        has_img = False
        has_tag = False
        imgs_paths = []
        
        for file in os.listdir(sample_folder):
            name, extension = os.path.splitext(file)
            if extension == '.png':
                has_img = True
                imgs_paths.append(os.path.join(sample_folder, file))
            if file == 'tag.txt':
                has_tag = True
                
        sample_num = int(os.path.basename(sample_folder))
        fabric_folder_name = os.path.basename(os.path.dirname(sample_folder))
        if not (has_img and has_tag):
            raise Exception(f"sample {sample_num} in {fabric_folder_name} is missing tag or images")
        
        # print(f'{None if sample_num else }')
        return [sample_num, imgs_paths, fabric_folder_name.lower()]
       
    def comp_to_vector(self, comp):
        # v = [0 for _ in self.clothing_types]
        # for f in comp:
        #     i = self.clothing_types.index(f)
        #     v[i] = comp[f]
        # v = v/100
        return [comp.get(f, 0) / 100 for f in self.base_fabric_types]
     
def get_dataloaders(config):
    # initialize a seperate dataset for train/val
    train_dataset = FabricDataset(data_path=config['data_path'], config=config, training=True)
    val_dataset = FabricDataset(data_path=config['data_path'], config=config, training=False)
    
    # find train and val size
    train_size = int(len(train_dataset) * 0.9)
    val_size = len(train_dataset) - train_size

    #split
    train_dataset, _ = random_split(train_dataset, [train_size, val_size])
    _, val_dataset = random_split(val_dataset, [train_size, val_size])

    # initialize dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    return train_dataloader, val_dataloader

