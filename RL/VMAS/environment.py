import vmas

def get_env(device):
    return vmas.make_env(
        scenario="balance", # can be scenario name or BaseScenario class
        num_envs=1,
        device=device, # Or "cuda" for GPU
        continuous_actions=False,
        wrapper=None,  # One of: None, "rllib", "gym", "gymnasium", "gymnasium_vec"
        max_steps=None, # Defines the horizon. None is infinite horizon.
        seed=None, # Seed of the environment
        dict_spaces=False, # By default tuple spaces are used with each element in the tuple being an agent.
        # If dict_spaces=True, the spaces will become Dict with each key being the agent's name
        grad_enabled=False, # If grad_enabled the simulator is differentiable and gradients can flow from output to input
        terminated_truncated=False, # If terminated_truncated the simulator will return separate `terminated` and `truncated` flags in the `done()`, `step()`, and `get_from_scenario()` functions instead of a single `done` flag
        n_agents = 3,
        # **kwargs # Additional arguments you want to pass to the scenario initialization
    )
