from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config
import torch
import torch.nn as nn
import torch.nn
from timeStamp import TimeStamp
from models.ValueEstimationModel import ValueEstimation
from rewardCalculator import RewardCalculator
from advantageEstimator import AdvantageEstimator
from conjugateGradientOptim import ConjugateGradientOptim
from torch.optim.lr_scheduler import LambdaLR

MODEL_PATH = "policy.pt"
VALUE_PATH = "value_estimation.pt"

def warmup_scheduler(step, warmup_steps=100):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    return 1.0



def memory(): #for debugging
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    #print(f"free{f * 1e-9}")
    #print(f"allocated{a * 1e-9}")
    #print(f"reserved{r * 1e-9}")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class APAAgent:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-large")
        config = GPT2Config.from_pretrained("openai-community/gpt2-large")
        self.model = GPT2LMHeadModel(config)
        self.model_init = GPT2LMHeadModel.from_pretrained("openai-community/gpt2-large")
        self.valueModel = ValueEstimation()
        self.rewardCalculator = RewardCalculator()
        self.advantageEstimator = AdvantageEstimator()

    def from_pretrained(self):
        self.model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2-large")
        self.valueModel.from_pretrained()
    
    def setup_training(self):
        self.model.train()
        self.model.to(device)
        self.model_init.to(device)
        self.valueModel.train()
        self.valueModel.to(device)
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=0.000001)
        self.scheduler = LambdaLR(self.optim, lr_lambda=warmup_scheduler)

    def load(self):
        self.valueModel.load_state_dict(torch.load(VALUE_PATH, weights_only=True))
        self.valueModel.from_pretrained()
        self.model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

    def save(self):
        torch.save(self.valueModel.state_dict(), VALUE_PATH)
        torch.save(self.model.state_dict(), MODEL_PATH)

    def train(self, load=False, batch_size=100, epochs=100):
        if load:
            self.load()
        else:
            self.from_pretrained()

        self.setup_training()
        for epoch in range(epochs):
            for i in range(batch_size):
                self.rollout(i)
            self.save()


    def rollout(self, i=0):
        memory()
        self.timeStamps = []
        self.optim.zero_grad()
        state, selected_token, last_token_probs, init_last_token_probs, done = self.generate_next_token()
        probs = last_token_probs.unsqueeze(0)
        init_probs = init_last_token_probs.unsqueeze(0)
        
        print(f"index{i}")
        max_tokens = 20
        while not done:
            memory()
            old_state = state
            state, selected_token, last_token_probs, init_last_token_probs, done = self.generate_next_token(state=state)
            self.timeStamps.append(TimeStamp(state=old_state, action=selected_token, reward=0, done=done))
            if len(self.timeStamps) >= max_tokens:
                done=True
            if not done:
                probs = torch.cat([probs, last_token_probs.unsqueeze(0)])
                init_probs = torch.cat([init_probs, init_last_token_probs.unsqueeze(0)])

        states = [timeStamp.state for timeStamp in self.timeStamps]
        try:
            reward = self.rewardCalculator.get_reward(states[-1])
            print(states[-1])
            print(f"reward{reward}")
        except:
            print(states)
            return
        self.timeStamps[-1].reward = reward
        values = self.get_discounted_future_reward(reward, len(states))
        est_values = self.valueModel(states)

        for i, timeStamp in enumerate(self.timeStamps):
            timeStamp.value_est = est_values[i]

        advantages = self.advantageEstimator.calc_advantages(self.timeStamps).to(device)
        apa_loss = self.get_apa_loss(probs, init_probs, advantages)
        apa_loss.backward()
        #if (i % 10) == 0:
        print(f"lossAPA{apa_loss.item()}")
        self.optim.step()
        self.scheduler.step()

        memory()
        est_values = self.valueModel(states)
        value_loss = self.get_value_function_loss(est_values, values)
        #if (i % 10) == 0:
        print(f"lossV{value_loss.item()}")
        optim = ConjugateGradientOptim()
        
        optim.backward(self.valueModel, value_loss)
        self.model.zero_grad()
        memory()

    def get_value_function_loss(self, predicted_values, target_values):
        loss_fn = nn.MSELoss()
        loss = loss_fn(predicted_values, target_values)
        return loss

    def get_apa_loss(self, probs, init_probs, advantages, lam=10):
        probs = torch.log(probs)
        init_probs = torch.log(init_probs)
        print(advantages)
        advantages = advantages / lam
        apa_loss = probs - advantages - init_probs
        apa_loss = apa_loss.pow(2)
        apa_loss = apa_loss.mean()
        return apa_loss

    def get_discounted_future_reward(self, reward, length, discount_factor=.99):
        values = torch.ones(length).to(device)
        cumulative_reward = reward
        for i in range(length - 1, -1, -1):
            values[i] = cumulative_reward
            cumulative_reward = cumulative_reward * discount_factor

        return values

    def generate_next_token(self, state=None, k=5):
        output = None
        init_output = None
        if state != None:
            input = self.tokenizer(state, return_tensors="pt")
            output = self.model(input_ids=input.input_ids.to(device), attention_mask=input.attention_mask.to(device))[0]
            init_output = self.model_init(input_ids=input.input_ids.to(device), attention_mask=input.attention_mask.to(device))[0]
        else:
            output = self.model(input_ids=torch.tensor([[self.tokenizer.eos_token_id]]).to(device))[0]
            init_output = self.model_init(input_ids=torch.tensor([[self.tokenizer.eos_token_id]]).to(device))[0]
        last_token_logits = output[0, -1]
        init_last_token_logits = init_output[0, -1]
        last_token_probs = nn.functional.softmax(last_token_logits, dim=0)
        init_last_token_probs = nn.functional.softmax(init_last_token_logits, dim=0)
        top_k = torch.topk(last_token_probs, 5)
        selected_indx = torch.multinomial(top_k.values, k)[0]
        selected_token = top_k.indices[selected_indx]
        #selected_token = torch.argmax(last_token_probs).item()

        eos_prob = last_token_probs[self.tokenizer.eos_token_id].item()
        if eos_prob < .001:
            input_ids = torch.cat([input["input_ids"].squeeze(0), torch.tensor([selected_token])]) if state != None else torch.tensor([selected_token])
            state = self.tokenizer.decode(input_ids)
        
        return state, selected_token, last_token_probs[selected_token], init_last_token_probs[selected_token], eos_prob > .001

x = APAAgent()
x.train()