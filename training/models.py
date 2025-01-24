import torch.nn as nn
from transformers import BertModel
from torchvision import models as vision_models
import torch

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Freeze the BERT model (so its not trainable but useable still)
        for param in self.bert.parameters():
            param.requires_grad = False

        # BERT Outputs 768 which is then converted to 128 to make summary smaller
        self.projection = nn.Linear(768, 128)

    def forward(self, input_ids, attention_mask):
        # Extract BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation
        pooler_output = outputs.pooler_output

        return self.projection(pooler_output)

class VideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Get the ResNet 3D 18 layers pretrained model
        self.backbone = vision_models.video.r3d_18(pretrained=True)

        # Sets all layers to not be training
        for param in self.backbone.parameters():
            param.requires_grad = False

        num_fts = self.backbone.fc.in_features

        # Defines the fc (classification head) so only the linear layer is trainable
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_fts, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    
    def forward(self, x):
        # x represents the entire tensor that contains all the video frames
        # [batch_size, frames, channels, height, width] -> [batch_size, channels, frames, height, width]
        x = x.transpose(1, 2)
        return self.backbone(x)

class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # Lower level features
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Higher level features
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        for param in self.conv_layers.parameters():
            param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        # [batch_size, 1, 64 (frequencies (y-axis)), 300 (time (x-axis))] -> [batch_size, 64, 300]
        x = x.squeeze(1)

        # Run through conv layers
        features = self.conv_layers(x)

        # Features output: [batch_size, 128, 1]

        return self.projection(features.squeeze(-1))
