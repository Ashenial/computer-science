import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import vmas.simulator.environment as environment
from model import ActorNetwork, CriticNetwork
from Buffer import Buffer

class MAPPO:
    def __init__(self, num_agents, env: environment.Environment, device, lr=3e-4, gamma=0.99, epsilon=0.2, lam=0.9):
        self.num_agents = num_agents
        self.env = env
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon
        self.device = device
        # self.entropy_coef = 0.01
        self.K_epoch = 4

        # 定义去中心化的Actor和集中式Critic
        # print(env.observation_space[0].shape[0], ' ', env.action_space[0].n) # 16, 9; state:(agents, env, state)(4, 10, 16)
        self.old_actors = [ActorNetwork(env.observation_space[agent_idx].shape[0], env.action_space[agent_idx].n).to(self.device) for agent_idx in range(num_agents)]
        self.actors = [ActorNetwork(env.observation_space[agent_idx].shape[0], env.action_space[agent_idx].n).to(self.device) for agent_idx in range(num_agents)]
        for old_actor, actor in zip(self.old_actors, self.actors):
            old_actor.load_state_dict(actor.state_dict())
        self.critic = CriticNetwork(env.observation_space[0].shape[0]).to(self.device)  # 集中的Critic
        self.actor_optimizers = [torch.optim.Adam(self.actors[agent_idx].parameters(), lr=lr, weight_decay=1e-5) for agent_idx in range(num_agents)]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr, weight_decay=1e-5)  # 集中式Critic的优化器
        self.buffers = [Buffer(self.device) for _ in range(num_agents)]

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
        actions = []
        old_probs = []
        for agent_idx, states_of_agent in enumerate(states): 
            agent_actions = []  
            agent_old_probs = []  
            for env_idx, state in enumerate(states_of_agent): # state.shape:(observation_space,) = (16,)
                state = state.to(self.device)  
                with torch.no_grad():
                    action_probs = self.old_actors[agent_idx](state)  # 获取动作的概率分布, shape: (action_space,) = (9,)
                    # 从概率分布中选择一个动作
                    action = action_probs.multinomial(1).squeeze()  # shape: (action_size_of_agent,) = (1,)  
                    action_prob = action_probs.gather(0, action.unsqueeze(-1))  # 获取该动作的概率

                agent_actions.append(action)  # agent_actions.shape:(num_envs, action_size_of_agent)
                agent_old_probs.append(action_prob.squeeze()) 

            actions.append(torch.stack(agent_actions))  # actions.shape:(num_agents,(num_envs, action_size_of_agent))
            old_probs.append(torch.stack(agent_old_probs))   
        return actions, old_probs

    def update(self):
        for _ in range(self.K_epoch):
            for old_actor, actor in zip(self.old_actors, self.actors):
                old_actor.load_state_dict(actor.state_dict())
                
            for agent_idx in range(self.num_agents):
                buffer = self.buffers[agent_idx]
                states_, actions_, old_probs_, rewards_, dones_, next_states_ = buffer.load()

                values = self.critic(states_)
                next_values = self.critic(next_states_)
                returns, advantages = self.compute_advantage(rewards_, values, next_values, dones_)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

                action_prob = self.actors[agent_idx](states_).gather(1, actions_.long().unsqueeze(-1))
                ratio = action_prob / old_probs_ # 计算新旧策略的比率
                clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) #clip 

                actor_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
                critic_loss = torch.mean(nn.MSELoss()(values, returns))
                # print(f"actor_loss: {actor_loss}, critic_loss: {critic_loss}")

                # BP
                self.actor_optimizers[agent_idx].zero_grad()
                actor_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.actors[agent_idx].parameters(), 10.0) # 裁剪渐变以避免过大的更新
                self.actor_optimizers[agent_idx].step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
                self.critic_optimizer.step()

        for agent_idx in range(self.num_agents):
            buffer = self.buffers[agent_idx]
            buffer.clear()

    def train(self, episodes=400, num_steps=6000):
        env = self.env
        episode_rewards = []
        
        for episode in range(episodes):
            states = self.env.reset() # 重置env, shape: (num_agents, num_envs, state_dim)
            num_envs = env.num_envs
            episode_reward = np.zeros((num_envs, self.num_agents))
            update_steps = 2000 # 每隔update_steps步更新一次actor和critic网络

            for step in range(num_steps):
                actions, old_probs = self.select_actions(states)  # actions.shape: (num_agents, num_envs, action_size_of_agent)
                next_states, rewards, dones, infos = self.env.step(actions)
                dones = dones.to(dtype=torch.float32) 

                for agent_idx, buffer in enumerate(self.buffers):
                    for env_idx in range(num_envs):
                        buffer.store(
                            states[agent_idx][env_idx],
                            actions[agent_idx][env_idx],
                            old_probs[agent_idx][env_idx],
                            rewards[agent_idx][env_idx],
                            dones[env_idx],
                            next_states[agent_idx][env_idx],
                        )
                        episode_reward[env_idx, agent_idx] += rewards[agent_idx][env_idx]

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




                
                
