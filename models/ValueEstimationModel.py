from transformers import BartTokenizer, BartModel, BartConfig
import torch
import torch.nn as nn

BART_HIDDEN_SIZE = 768
LINEAR_HIDDEN_SIZE = 2000

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
        outputs = self.model(input_ids=inputs.input_ids.to(device), attention_mask=inputs.attention_mask.to(device))
        outputs = self.valueEstimationHead(outputs.last_hidden_state.mean(1))

        return outputs.squeeze(1)

"""
x = ValueEstimation()
x.from_pretrained()
for param in x.parameters():
    print(param)
"""