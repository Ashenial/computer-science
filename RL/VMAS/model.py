import torch
import torch.nn as nn

"""The model used for all critics and actors is 
a two layer Multi Layer Perceptron (MLP) with 
hyperbolic tangent activations"""

hidden_dim = 128

class ActorNetwork(nn.Module): # Policy
    def __init__(self, input_dim, output_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = torch.tanh(self.fc1(x))  
        x = torch.tanh(self.fc2(x))  
        action_probs = torch.softmax(self.fc3(x), dim=-1)  # 输出动作的概率分布
        return action_probs

class CriticNetwork(nn.Module): # Value
    def __init__(self, input_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1) 

    def forward(self, x):
        x = torch.tanh(self.fc1(x))  
        x = torch.tanh(self.fc2(x))  
        value = self.fc3(x)  
        return value
