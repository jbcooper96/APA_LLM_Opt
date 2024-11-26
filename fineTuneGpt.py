from datasets import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel
import torch
from torch.utils.data import DataLoader


def memory(): #for debugging
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    m = torch.cuda.max_memory_allocated(0)
    f = r-a  # free inside reserved
    print(f"free{f * 1e-9}")
    print(f"allocated{a * 1e-9}")
    print(f"reserved{r * 1e-9}")
    print(f"max{m * 1e-9}")
    

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

ds = load_dataset("gxb912/large-twitter-tweets-sentiment")

dataset=ds["train"]
test=ds["test"]

model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2-large")
model.to(device)

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-large")
tokenizer.pad_token = tokenizer.eos_token

optim = torch.optim.AdamW(model.parameters(), lr=0.0001)

loss_fn = torch.nn.CrossEntropyLoss()

def tokenize_function(examples):
    return tokenizer(examples["text"], padding=True, truncation=True, max_length=150, return_tensors="pt")

dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names, batch_size=32)
test = test.map(tokenize_function, batched=True, remove_columns=test.column_names, batch_size=32)

dataset.set_format("torch")


dataloader = DataLoader(dataset, batch_size=32) 
testLoader = DataLoader(test, batch_size=32) 

epochs = 3

val_loss = []
for epoch in range(epochs):
    print(f"epoch{epoch+1}")
    model.train()
    try:
        for i, batch in enumerate(dataloader):
            optim.zero_grad()
            input_ids = batch["input_ids"].to(device)
            torch.cuda.empty_cache()
            output = model(input_ids=input_ids, attention_mask=batch["attention_mask"].to(device), labels=input_ids)
            
            output.loss.backward()
            optim.step()
            if i % 20 == 0:
                print(output.loss.item())
                #memory()
    except:
        print(torch.cuda.memory_summary())

    torch.save(model.state_dict(), "twitter-fine-tune.pt")
    model.eval()
    print("eval:")
    avg_loss = 0.0
    batch_count = 0
    for batch in testLoader:
        batch_count += 1
        output = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device), labels=batch["input_ids"].to(device))
        avg_loss += output.loss.item()

    avg_loss /= batch_count
    val_loss.append(avg_loss)
    print(avg_loss)

print(val_loss)

    


