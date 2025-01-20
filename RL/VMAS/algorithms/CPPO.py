import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import vmas.simulator.environment as environment
from model import ActorNetwork, CriticNetwork
from Buffer import Buffer

class CPPO:
    def __init__(self, num_agents, env: environment.Environment, device, lr=3e-4, gamma=0.99, epsilon=0.2, lam=0.9):
        self.num_agents = num_agents
        self.env = env
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon
        self.device = device
        self.K_epoch = 4

        # 定义集中的Actor和集中式Critic
        self.actor = ActorNetwork(env.observation_space[0].shape[0] * num_agents, env.action_space[0].n).to(self.device)
        self.old_actor = ActorNetwork(env.observation_space[0].shape[0] * num_agents, env.action_space[0].n).to(self.device)
        self.old_actor.load_state_dict(self.actor.state_dict())
        self.critic = CriticNetwork(env.observation_space[0].shape[0]).to(self.device)  
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, weight_decay=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr, weight_decay=1e-5)  
        self.buffer = Buffer(self.device)

    def compute_advantage(self, rewards, values, next_values, dones):
        deltas = rewards + self.gamma * next_values * (1 - dones) - values
        advantages = torch.zeros_like(deltas)
        adv = 0
        for t in reversed(range(len(deltas))):
            adv = deltas[t] + self.gamma * self.lam * adv
            advantages[t] = adv
        returns = advantages + values
        return returns, advantages

    def select_actions(self, states): 
        combined_states = torch.cat(states, dim=-1).to(self.device)  # Combine states from all agents
        actions = [torch.zeros(self.env.num_envs,1) for _ in range(self.num_agents)]
        action_probs = [torch.zeros(self.env.num_envs,1) for _ in range(self.num_agents)]
        with torch.no_grad():
            action_prob = self.old_actor(combined_states)  # 获取动作的概率分布
            action = action_prob.multinomial(1)  # 从概率分布中选择一个动作
            action_prob = action_prob.gather(1, action)  # 获取该动作的概率
        for agent_idx in range(self.num_agents):
                actions[agent_idx] = action
                action_probs[agent_idx] = action_prob
        return actions, action_probs
    
    def update(self):
        for _ in range(self.K_epoch):
            self.old_actor.load_state_dict(self.actor.state_dict())
            states_, actions_, old_probs_, rewards_, dones_, next_states_ = self.buffer.load()

            values = self.critic(states_)
            next_values = self.critic(next_states_)
            returns, advantages = self.compute_advantage(rewards_, values, next_values, dones_)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

            action_prob = self.actor(states_).gather(1, actions_.long().unsqueeze(-1))
            ratio = action_prob / old_probs_  # 计算新旧策略的比率
            clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)  # clip 

            actor_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            critic_loss = torch.mean(nn.MSELoss()(values, returns))

            # BP
            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)  # 裁剪渐变以避免过大的更新
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
            self.critic_optimizer.step()

        self.buffer.clear()

    def train(self, episodes=400, num_steps=6000):
        env = self.env
        episode_rewards = []
        
        for episode in range(episodes):
            states = self.env.reset()  # 重置env, shape: (num_agents, num_envs, state_dim)
            num_envs = env.num_envs
            episode_reward = 0
            update_steps = 2000  # 每隔update_steps步更新一次actor和critic网络

            for step in range(num_steps):
                actions, old_probs = self.select_actions(states)  # actions.shape: (num_agents, num_envs, action_size_of_agent)
                next_states, rewards, dones, infos = self.env.step(actions)
                dones = dones.to(dtype=torch.float32) 

                for agent_idx in range(self.num_agents):
                    for env_idx in range(num_envs):
                        self.buffer.store(
                            states[agent_idx][env_idx],
                            actions[agent_idx][env_idx],
                            old_probs[agent_idx][env_idx],
                            rewards[agent_idx][env_idx],
                            dones[env_idx],
                            next_states[agent_idx][env_idx],
                        )

                episode_reward += torch.mean(torch.stack(rewards)).item()

                states = next_states

                # if all envs have done, terminate now.
                if torch.all(dones):
                    break

                # update actor network and centralized critic network
                # if (step+1) % update_steps == 0:
                #     self.update()

            self.update()

            # 记录每个episode的平均奖励
            total_episode_reward = np.mean(episode_reward)  # 对每个智能体计算平均奖励
            episode_rewards.append(total_episode_reward)

            print(f"Episode {episode + 1}, Rewards: \n{episode_reward}")
        return episode_rewards



