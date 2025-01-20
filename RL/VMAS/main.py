import numpy as np
import torch
import matplotlib.pyplot as plt

import environment
from algorithms.MAPPO import MAPPO
from algorithms.IPPO import IPPO
from algorithms.CPPO import CPPO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

def main():
    env = environment.get_env(device=device)
    # action_space:(4, 9); observation_space:(4,16)
    num_agents = env.n_agents

    # MAPPO
    mappo = MAPPO(num_agents=num_agents, env=env, device=device)
    episode_rewards_mappo = mappo.train()

    # IPPO
    ippo = IPPO(num_agents=num_agents, env=env, device=device)
    episode_rewards_ippo = ippo.train()

    # CPPO
    cppo = CPPO(num_agents=num_agents, env=env, device=device)
    episode_rewards_cppo = cppo.train()

    episode_numbers = range(1, len(episode_rewards_mappo) + 1)
    plt.plot(episode_numbers, episode_rewards_mappo, color='green', label='MAPPO')  
    plt.plot(episode_numbers, episode_rewards_ippo, color='orange', label='IPPO')   
    plt.plot(episode_numbers, episode_rewards_cppo, color='blue', label='CPPO')    
    plt.legend()
    plt.xlabel("Training iteration")
    plt.ylabel("Episode reward mean")
    plt.title("Balance")
    plt.savefig("Balance.png")  # 保存为PNG格式
    plt.show()

if __name__ == "__main__":
    main()
