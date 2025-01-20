import torch
import random

class Buffer:
    def __init__(self, device="cpu"):
        self.states = []
        self.actions = []
        self.action_probs = []
        self.rewards = []
        self.dones = []
        self.next_states = []
        self.device = device

    def store(self, state, action, action_prob, reward, done, next_state):
        self.states.append(state)
        self.actions.append(action)
        self.action_probs.append(action_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)

    def load(self):
        states = torch.stack(self.states).to(self.device)
        actions = torch.tensor(self.actions).to(self.device)
        action_probs = torch.tensor(self.action_probs).view(-1,1).to(self.device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32).view(-1,1).to(self.device)
        dones = torch.tensor(self.dones, dtype=torch.float32).view(-1,1).to(self.device)
        next_states = torch.stack(self.next_states).to(self.device)
        return states, actions, action_probs, rewards, dones, next_states
    
    def is_empty(self):
        return len(self.states) == 0

    def sample(self, batch_size):
        """Randomly sample a batch of data from the memory."""
        if len(self.states) < batch_size:
            batch_size = len(self.states)

        indices = random.sample(range(len(self.states)), batch_size)
        sampled_states = [self.states[i] for i in indices]
        sampled_actions = [self.actions[i] for i in indices]
        sampled_action_probs = [self.action_probs[i] for i in indices]
        sampled_rewards = [self.rewards[i] for i in indices]
        sampled_dones = [self.dones[i] for i in indices]
        sampled_next_states = [self.next_states[i] for i in indices]

        return (
            torch.stack(sampled_states).to(self.device),
            torch.tensor(sampled_actions).to(self.device),
            torch.tensor(sampled_action_probs).view(-1,1).to(self.device),
            torch.tensor(sampled_rewards).view(-1,1).to(self.device),
            torch.tensor(sampled_dones).view(-1,1).to(self.device),
            torch.stack(sampled_next_states).to(self.device),
        )

    def clear(self):
        self.states = []
        self.actions = []
        self.action_probs = []
        self.rewards = []
        self.dones = []
        self.next_states = []
