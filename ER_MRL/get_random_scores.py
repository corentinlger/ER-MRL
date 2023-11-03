import numpy as np
import gymnasium as gym 
from stable_baselines3 import PPO


env_ids = ['Ant-v4',
           'HalfCheetah-v4',
           'Hopper-v4',
           'Humanoid-v4',
           'HumanoidStandup-v4',
           'InvertedDoublePendulum-v4',
           'InvertedPendulum-v4',
           'Pusher-v4',
           'Reacher-v4',
           'Swimmer-v4',
           'Walker2d-v4'
           ]

env_scores = []

logs_name = "logs_random_agents.txt"

for env_id in env_ids:
    # Evaluating Random_agent

    env = gym.make(env_id)
    nb_episodes = 100
    mean_ep_rewards = []

    for ep in range(nb_episodes):
        ep_rewards = []
        done = False
        truncated = False
        obs, info = env.reset() 

        while not done and not truncated:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            ep_rewards.append(reward)

        # end of ep:
        mean_ep_reward = np.mean(ep_rewards)
        mean_ep_rewards.append(mean_ep_reward)
    
    env_mean_reward = np.mean(mean_ep_rewards)
    env_scores.append(env_mean_reward)

with open(logs_name, "w") as f:    
    for i in range(len(env_ids)):
        print(f"env: {env_ids[i]} score: {env_scores[i]}\n")
        f.writelines(f"\nenv: {env_ids[i]} score: {env_scores[i]}")


    

