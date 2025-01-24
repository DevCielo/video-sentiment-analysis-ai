import torch.nn as nn
from transformers import BertModel
from torchvision import models as vision_models

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