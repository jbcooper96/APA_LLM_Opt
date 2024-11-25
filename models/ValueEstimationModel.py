from transformers import BartTokenizer, BartModel, BartConfig
import torch
import torch.nn as nn

BART_HIDDEN_SIZE = 768
LINEAR_HIDDEN_SIZE = 2000

class ValueEstimation(nn.Module):
    def __init__(self):
        super().__init__()
        config = BartConfig.from_pretrained('facebook/bart-base')
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        self.model = BartModel(config)
        self.valueEstimationHead = nn.Sequential(
            nn.Linear(BART_HIDDEN_SIZE, LINEAR_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(LINEAR_HIDDEN_SIZE, 1),
            nn.Sigmoid()
        )
        
    
    def from_pretrained(self):
        self.model = BartModel.from_pretrained('facebook/bart-base')
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        outputs = self.valueEstimationHead(outputs.last_hidden_state.mean(1))

        return outputs.squeeze(1)

"""
x = ValueEstimation()
x.from_pretrained()
for param in x.parameters():
    print(param)
"""