import gym
import torch
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=-1)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, lam=0.95, eps_clip=0.2, K_epochs=4):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.policy_old = PolicyNetwork(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_network = ValueNetwork(state_dim)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            # action_probs = self.policy_old(state)
            action_probs = self.policy(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item()
    
    def compute_gae(self, rewards, dones, values, next_value):
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lam * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            next_value = values[i]
        returns = [adv + val for adv, val in zip(advantages, values)]
        return returns, advantages

    def update(self, memory): # GAE
        states = torch.FloatTensor(memory['states'])
        actions = torch.LongTensor(memory['actions'])
        logprobs = torch.FloatTensor(memory['logprobs'])
        rewards = torch.FloatTensor(memory['rewards'])
        dones = torch.FloatTensor(memory['dones'])

        values = self.value_network(states).detach().squeeze()
        next_value = self.value_network(states[-1].unsqueeze(0)).detach().squeeze()
        returns, advantages = self.compute_gae(rewards, dones, values, next_value)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)

        for _ in range(self.K_epochs):
            self.policy_old.load_state_dict(self.policy.state_dict())

            # Optimize policy network (actor)
            action_probs = self.policy(states)
            dist = Categorical(action_probs)
            new_logprobs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratios = torch.exp(new_logprobs - logprobs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2) - 0.01 * entropy

            self.optimizer.zero_grad()
            actor_loss.mean().backward(retain_graph=True)
            self.optimizer.step()

            # Optimize value network (critic)
            state_values = self.value_network(states).squeeze()
            critic_loss = self.MseLoss(state_values, returns)

            self.value_optimizer.zero_grad()
            critic_loss.backward()
            self.value_optimizer.step()

            # print(f"actor_loss: {actor_loss.mean()}, critic_loss: {critic_loss}")

        

    # def update(self, memory):
    #     states = torch.FloatTensor(memory['states'])
    #     actions = torch.LongTensor(memory['actions'])
    #     logprobs = torch.FloatTensor(memory['logprobs'])
    #     rewards = torch.FloatTensor(memory['rewards'])
    #     dones = torch.FloatTensor(memory['dones'])

    #     returns = []
    #     discounted_reward = 0
    #     for reward, done in zip(reversed(rewards), reversed(dones)):
    #         if done:
    #             discounted_reward = 0
    #         discounted_reward = reward + (self.gamma * discounted_reward)
    #         returns.insert(0, discounted_reward)
    #     returns = torch.FloatTensor(returns)

    #     for _ in range(self.K_epochs):
    #         # Optimize policy network (actor)
    #         action_probs = self.policy(states)
    #         dist = Categorical(action_probs)
    #         new_logprobs = dist.log_prob(actions)
    #         entropy = dist.entropy().mean()

    #         ratios = torch.exp(new_logprobs - logprobs)
    #         advantages = returns - self.value_network(states).detach().squeeze()
    #         surr1 = ratios * advantages
    #         surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
    #         actor_loss = -torch.min(surr1, surr2) - 0.01 * entropy

    #         self.optimizer.zero_grad()
    #         actor_loss.mean().backward(retain_graph=True)
    #         self.optimizer.step()

    #         # Optimize value network (critic)
    #         state_values = self.value_network(states).squeeze()
    #         critic_loss = self.MseLoss(state_values, returns)

    #         self.value_optimizer.zero_grad()
    #         critic_loss.backward()
    #         self.value_optimizer.step()

    #     self.policy_old.load_state_dict(self.policy.state_dict())

def main():
    rewards_per_episode = []

    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    ppo = PPO(state_dim, action_dim)
    max_episodes = 500
    max_timesteps = 600
    update_timestep = 2000
    memory = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'dones': []}
    timestep = 0

    for episode in range(max_episodes):
        state, _ = env.reset()
        total_reward = 0
        for t in range(max_timesteps):
            timestep += 1
            action, logprob = ppo.select_action(state)
            next_state, reward, t1, t2, _ = env.step(action)
            done = t1 or t2
            memory['states'].append(state)
            memory['actions'].append(action)
            memory['logprobs'].append(logprob)
            memory['rewards'].append(reward)
            memory['dones'].append(done)
            state = next_state
            total_reward += reward

            if done:
                break

        ppo.update(memory)
        memory = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'dones': []}
        timestep = 0

        rewards_per_episode.append(total_reward)
        print(f"Episode {episode+1} finished with reward: {total_reward}")

    plt.plot(rewards_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.show()


if __name__ == '__main__':
    main()