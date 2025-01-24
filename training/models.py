import torch.nn as nn
from transformers import BertModel

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Freeze the BERT model (so its not trainable but useable still)
        for param in self.bert.parameters():
            param.requires_grad = False

        # Outputs 768 which is then converted to 128 to make summary smaller
        self.projection = nn.Linear(768, 128)

    def forward(self, input_ids, attention_mask):
        # Extract BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation
        pooler_output = outputs.pooler_output

        return self.projection(pooler_output)


