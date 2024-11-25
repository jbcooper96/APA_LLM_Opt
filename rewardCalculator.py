from transformers import pipeline

class RewardCalculator:
    def __init__(self, emotion="anger"):
        self.classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
        self.emotion = emotion

    def get_reward(self, state):
        model_outputs = self.classifier([state])
        for emotion in model_outputs[0]:
            if emotion["label"] == self.emotion:
                return emotion["score"]

        print("Error finding emotion")
        return 0
        
