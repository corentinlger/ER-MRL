import os
import glob
import shutil
import numpy as np
import pandas as pd 
import gymnasium as gym
import matplotlib.pyplot as plt 

import tensorflow as tf 
from tensorflow.python.summary.summary_iterator import summary_iterator

def get_logs_values(logdir, models, nb_seeds):
    """
    Download data from Tensorboard log files and returns the mean episodic reward array, the std episodic reward array 
    and the timesteps array associated in order to plot and observe these results for the desired models.
    
    :param logdir: (str) log directory
    :param models: ([str]) models names list 
    :return: (np.ndarray, np.ndarray, np.ndarray) models_mean_arrays, models_std_arrays, timesteps
    """
    
    # Downloading the data from the logs directory in a pandas DataFrame
    dfs = []
    timesteps = {}
    
    for dir in os.listdir(logdir):
        if os.path.isdir(os.path.join(logdir, dir)):
            for model_name in models :
                if dir.startswith(model_name) and dir[len(model_name) + 1].isdigit():
                    model = model_name
                    for i in range(nb_seeds):
                        if dir.endswith(f"{i}_1"):
                            seed = i

                            subdir = os.path.join(logdir, dir)
                            files = [os.path.join(subdir, file) for file in os.listdir(subdir)]
                            data = []

                            for filepath in files:
                                for event in summary_iterator(filepath):
                                    for value in event.summary.value:
                                        if value.tag =='rollout/ep_rew_mean':
                                            data.append([event.step, value.simple_value, model, seed])

                            df = pd.DataFrame(data, columns=['step', 'ep_rew_mean', 'model', 'seed'])
                            dfs.append(df)
                            
                            timesteps[model] = np.array(df.step, dtype=np.float64)

    df_all = pd.concat(dfs, ignore_index=True)

    
    #Transforming the pandas data into numpy arrays of mean and std reward for each model 

    models_tot_arrays = {}
    models_mean_arrays = {}
    models_std_arrays = {}

    for model in models:
        arr = []
        for seed in range(nb_seeds): 
            serie = df_all.loc[(df_all['model'] == model) & (df_all['seed'] == seed), 'ep_rew_mean']
            serie = serie.reset_index(drop=True)  # reset index
            if len(serie) == 0 : 
                print(f"data missing for {model} seed {seed}")
            else:
                arr.append(serie)
        
        # concatenate the results of all the models and put them in an array
        df = pd.concat(arr, axis=1)
        nparr = df.to_numpy()
        models_tot_arrays[model] = nparr

        # calculate the mean and the std of this model rewards:
        models_mean_arrays[model] = np.mean(nparr, axis=1)
        models_std_arrays[model] = np.std(nparr, axis=1) 

    return models_mean_arrays, models_std_arrays, timesteps



def plot_results(exp_name, logdir, models, nb_seeds, max_reward=None):
    models_mean_arrays, models_std_arrays, timesteps = get_logs_values(logdir, models, nb_seeds)

    plt.figure(figsize=(15,6))
    plt.title(f"Evolution of mean episode reward on '{exp_name}'")
    plt.xlabel('Steps')
    plt.ylabel('Mean episode reward')

    for model in models :
        plt.plot(timesteps[model], models_mean_arrays[model], label=model)
        plt.fill_between(timesteps[model], models_mean_arrays[model]-models_std_arrays[model], models_mean_arrays[model]+models_std_arrays[model], alpha=0.2)

    if max_reward:  
        plt.plot(timesteps[model], np.full(timesteps[model].shape[0], max_reward), color='black', linestyle='dashed', alpha=0.5, label='Max reward')
        
    plt.legend()
    plt.show()
    

def delete_logs_models(exp_name):
    logdir = f"logs/{exp_name}/"
    models_dir = f"models/{exp_name}/"
    
    confirmation = input(f"Are you sure you want to delete the directories for {exp_name} ? (Y/N)")
    if confirmation == 'Y':
        shutil.rmtree(logdir)
        shutil.rmtree(models_dir)
        print("directories deleted")
    else:
        pass


def random_agent_avg_reward(env_id, n_episodes):
    env = gym.make(env_id)    
    mean_ep_rewards = []

    for ep in range(n_episodes):
        env.reset()
        ep_reward = 0
        done = False
        truncated = False
        while not done and not truncated:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            
        mean_ep_rewards.append(ep_reward)

    return np.mean(mean_ep_rewards)