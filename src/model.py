import torch
import torch.nn as nn
from torchvision import models

class FabricCompositionPredictor(nn.Module):
    """
    Predicts fabric composition using both image features and clothing type embeddings.

    Args:
        feature_extraction_block (nn.Module): The block for extracting image features (e.g., ResNet50-based).
        clothing_embedding_block (nn.Module): The block for embedding clothing types.
        fusion_block (nn.Module): The block for fusing image and clothing embeddings.
        prediction_block (nn.Module): The block for predicting fabric composition from fused features.
    """
    def __init__(self, feature_extraction_block, attention_block, clothing_embedding_block, fusion_block, prediction_block, threshold_activation):
        super().__init__()
        
        self.feb = feature_extraction_block  # Feature extraction block
        self.attention = attention_block # sample attention block
        self.ceb = clothing_embedding_block  # Clothing embedding block
        self.fuse = fusion_block  # Fusion block
        self.pred = prediction_block  # Prediction block
        
        # ignore this field
        self.threshold_activation = threshold_activation 
        
        self.scale = nn.Parameter(torch.ones(1))
        self.project_residual = nn.Linear(1024, 28)
        
        
    def forward(self, i, c):
        """
        Forward pass of the model.

        Args:
            i (torch.Tensor): Input image tensor of shape (batch_size, 4, 3, 224, 224).
            c (torch.Tensor): One-hot encoded clothing type tensor of shape (batch_size, num_clothes).

        Returns:
            torch.Tensor: Predicted fabric composition tensor of shape (batch_size, num_fabrics).
        """
        # Extract image features
        feature_extractor_output = self.feb(i)
        
        # Self attention on image samples
        attention_output = self.attention(feature_extractor_output)
        
        # Embed clothing types
        clothing_embedding_output = self.ceb(c)
        
        # kind of a residual connection
        rand_img = torch.randint(0, 4, (1,)).item()
        
        feature_extractor_output = feature_extractor_output[:, rand_img, :] 
        
        # Fuse features
        fusion = self.fuse(attention_output, clothing_embedding_output) + attention_output * self.scale
        
        # Predict fabric composition
        prediction = self.pred(fusion)
        
        return prediction
        
        

class FeatureExtractor(nn.Module):
    """
    A feature extraction block based on a pre-trained ResNet-50 backbone.

    Args:
        fine_tune (bool): Whether to fine-tune the ResNet backbone or freeze its layers.
    """
    def __init__(self, feature_dim, dropout, fine_tune=False):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)  # Pretrained ResNet-50 backbone
        
        # Replace the classification head with identity to extract raw features
        self.backbone.fc = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(2048, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        if not fine_tune:
            # Freeze all layers
            for param in self.backbone.parameters():
                param.requires_grad = False
            
        # Unfreeze the final few layers (layer4 and layer3) for fine-tuning
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True
    
        for param in self.backbone.layer3.parameters():
            param.requires_grad = True
        
        for param in self.backbone.layer2.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        """
        Process multiple images and return their features.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_images, 3, 224, 224).

        Returns:
            torch.Tensor: Extracted features of shape (batch_size, num_images, feature_dim).
        """
        
        
        batch_size, num_images, c, h, w = x.shape
        x = x.view(batch_size * num_images, 3, 224, 224)  # Flatten to a single batch
        features = self.backbone(x)  # ResNet processes each image
        projected_features = self.projector(features)
        projected_features = projected_features.view(batch_size, num_images, -1)  # Reshape back
        return projected_features
    

class SingleHeadAttention(nn.Module):
    def __init__(self, feature_dim, dropout=0.1):
        super().__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)

        self.out = nn.Linear(feature_dim, feature_dim)

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)  # Attention along the feature dimension
        self.layer_norm = nn.LayerNorm(feature_dim)
        
         # Learned pooling weights for num_images
        self.pool_weights = nn.Parameter(torch.ones(4))
        self.pool_softmax = nn.Softmax(dim=-1)  # To normalize pooling weights

    def forward(self, image_features):
        """
        Args:
            image_features (torch.Tensor): Shape (batch_size, num_images, feature_dim)
        Returns:
            torch.Tensor: Fused feature tensor of shape (batch_size, feature_dim).
        """
        batch_size, num_images, feature_dim = image_features.shape

        # Compute queries, keys, and values
        queries = self.query(image_features)  # (batch_size, num_images, feature_dim)
        keys = self.key(image_features)      # (batch_size, num_images, feature_dim)
        values = self.value(image_features)  # (batch_size, num_images, feature_dim)

        # Compute attention scores and weights
        attention_scores = torch.matmul(queries, keys.transpose(-1, -2))  # (batch_size, num_images, num_images)
        attention_scores = attention_scores / (feature_dim ** 0.5)  # Scale by sqrt(feature_dim)
        attention_weights = self.softmax(attention_scores)  # (batch_size, num_images, num_images)

        # Aggregate the values
        attended_values = torch.matmul(attention_weights, values)  # (batch_size, num_images, feature_dim)

        # Learned pooling over num_images
        normalized_weights = self.pool_softmax(self.pool_weights)  # (num_images,)
        fused_features = torch.einsum('bni,n->bi', attended_values, normalized_weights)  # (batch_size, feature_dim)

        # Residual connection and normalization
        residual = image_features.mean(dim=1)  # (batch_size, feature_dim)
        fused_features = self.dropout(self.layer_norm(fused_features + residual))  # (batch_size, feature_dim)

        return fused_features




class ClothingEmbedding(nn.Module):
    """
    A clothing embedding block that maps one-hot clothing types to a dense feature representation.

    Args:
        num_clothes (int): Number of unique clothing types.
        feature_dim (int): Dimensionality of the output feature vector (default: 2048).
    """
    def __init__(self, num_clothes, feature_dim=2048):
        super().__init__()
        
        self.pipeline = nn.Sequential(
            nn.Linear(num_clothes, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
            nn.Dropout(0.4),
        )
            
    def forward(self, x):
        """
        Forward pass through the clothing embedding block.

        Args:
            x (torch.Tensor): One-hot encoded clothing type tensor of shape (batch_size, num_clothes).

        Returns:
            torch.Tensor: Clothing embedding tensor of shape (batch_size, feature_dim).
        """
        return self.pipeline(x)
        

class FusionLayer(nn.Module):
    """
    A fusion block based on Feature-wise Linear Modulation (FiLM) to combine image features and clothing embeddings.

    Args:
        feature_dim (int): Dimensionality of the feature vectors (e.g., 2048).
        dropout (float): Dropout probability for regularization (default: 0.05).
    """
    def __init__(self, feature_dim, dropout=0.05):
        super().__init__()
        self.scale = nn.Linear(feature_dim, feature_dim)  # Computes scaling factor gamma
        self.shift = nn.Linear(feature_dim, feature_dim)  # Computes shifting factor beta
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()  # Optional dropout
        self.norm = nn.LayerNorm(feature_dim)  # Layer normalization for stability
        
    def forward(self, feature_extractor_output, clothing_embedding_output):
        """
        Forward pass through the fusion block.

        Args:
            feature_extractor_output (torch.Tensor): Image features of shape (batch_size, feature_dim).
            clothing_embedding_output (torch.Tensor): Clothing embedding of shape (batch_size, feature_dim).

        Returns:
            torch.Tensor: Fused feature tensor of shape (batch_size, feature_dim).
        """
        # Compute FiLM modulation
        modulation = self.scale(clothing_embedding_output) * feature_extractor_output + self.shift(clothing_embedding_output)
        # Apply normalization and dropout
        residual = clothing_embedding_output  # Residual connection
        return self.norm(self.dropout(modulation + residual))  # Add residual


class FeedForwardPrediction(nn.Module):
    """
    A prediction block that maps fused features to fabric composition probabilities.

    Args:
        feature_dim (int): Input dimensionality (e.g., 2048).
        higher_dim (int): Hidden layer dimensionality for intermediate representations.
        num_fabrics (int): Number of output fabric types.
        dropout (float): Dropout probability for regularization (default: 0.05).
    """
    def __init__(self, feature_dim, higher_dim, num_fabrics, dropout=0.05):
        super().__init__()
        self.pipeline = nn.Sequential(
            nn.Linear(feature_dim, higher_dim),  # First hidden layer
            nn.ReLU(),  # Non-linearity
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),  # Dropout
            nn.Linear(higher_dim, feature_dim),  # Second hidden layer
            nn.ReLU(),  # Non-linearity
            nn.Linear(feature_dim, num_fabrics),  # Output layer
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Forward pass through the prediction block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, feature_dim).

        Returns:
            torch.Tensor: Predicted fabric composition tensor of shape (batch_size, num_fabrics).
        """
        return self.pipeline(x)

class ThresholdActivation(nn.Module):
    def __init__(self, threshold=0.1):
        super(ThresholdActivation, self).__init__()
        self.threshold = threshold

    def forward(self, x):
        # Apply threshold: Set values below the threshold to 0
        return torch.where(x >= self.threshold, x, torch.tensor(0.0, device=x.device))
    
    
def build_model(feature_dim, higher_dim, num_clothes, num_fabrics, dropout, threshold):
    
    # initialize all necessary blocks with their respective arguments
    feature_extraction_block = FeatureExtractor(feature_dim=feature_dim, dropout=dropout)
    attention_block = SingleHeadAttention(feature_dim=feature_dim, dropout=dropout)
    clothing_embedding_block = ClothingEmbedding(num_clothes=num_clothes, feature_dim=feature_dim)
    fusion_block = FusionLayer(feature_dim=feature_dim, dropout=dropout)
    prediction_feed_forward_layer = FeedForwardPrediction(feature_dim=feature_dim, higher_dim=higher_dim, num_fabrics=num_fabrics, dropout=dropout)
    threshold_actication_layer = ThresholdActivation(threshold)
    
    
    return FabricCompositionPredictor(feature_extraction_block, 
                                      attention_block, 
                                      clothing_embedding_block, 
                                      fusion_block, 
                                      prediction_feed_forward_layer,
                                      threshold_actication_layer)

def get_model(config):
    return build_model(feature_dim=config['feature_dim'],
                       higher_dim=config['higher_dim'],
                       num_clothes=config['num_clothes'],
                       num_fabrics=config['num_fabrics'],
                       dropout=config['dropout'],
                       threshold=config['threshold']
                )