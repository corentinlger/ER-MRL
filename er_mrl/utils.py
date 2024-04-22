import os
import shutil

import numpy as np
import pandas as pd 
import gymnasium as gym
from scipy.fft import fft
import matplotlib.pyplot as plt 
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


def plot_episode_obs(dones, obs_history, obs_idx_names, context_history, episode_id, ep_timesteps=None, legend=True, figsize=(8, 4)):
    ep_step_start = dones[episode_id - 1] if episode_id > 1 else 0 

    ep_step_stop = dones[episode_id]
    if ep_timesteps:
        nb_steps = ep_timesteps
    else:
        nb_steps = ep_step_stop - ep_step_start

    timesteps = np.arange(stop=nb_steps)

    """Plot the RL agents Observation """

    plt.figure(figsize=figsize)
    # Remove comment below if also wanna plot the actions
    #plt.plot(timesteps, actions_history[ep_step_start:ep_step_stop], label=action, alpha=0.5)

    for idx in range(len(obs_idx_names)):
        if ep_timesteps:
            observation = obs_history[ep_step_start:ep_step_start+ep_timesteps, idx]
        else:
            observation = obs_history[ep_step_start:ep_step_stop, idx]
        plt.plot(timesteps, observation, label=obs_idx_names[idx])
        plt.title(f"Observations o_t Episode {episode_id}")
        if legend:
            plt.legend()
    
    plt.savefig(f"figures/obs_ep_{episode_id}")
    plt.show()

    """Plot the ER-MRL agents Context"""

    plt.figure(figsize=figsize)
    # Remove comment below if also wanna plot the actions
    #plt.plot(timesteps, actions_history[ep_step_start:ep_step_stop], label=action, alpha=0.5)

    obs_RES_neurons = 20
    for idx in range(obs_RES_neurons):
        if ep_timesteps:
            context = context_history[ep_step_start:ep_step_start+ep_timesteps, idx]
        else:
            context = context_history[ep_step_start:ep_step_stop, idx]
        plt.plot(timesteps, context)
        plt.title(f"Contexts c_t Episode {episode_id}")
    
    plt.savefig(f"figures/ctx_ep_{episode_id}")
    plt.show()

    ep_step_start += (ep_step_stop - ep_step_start)
    

def plot_episode_fft(dones, obs_history, context_history, episode_id):

    ep_step_start = dones[episode_id - 1] if episode_id > 1 else 0 

    ep_step_stop = dones[episode_id]
    nb_steps = ep_step_stop - ep_step_start
    timesteps = np.arange(stop=nb_steps)

    """Plot the fft of RL agents observations"""

    observation = obs_history[ep_step_start:ep_step_stop, :]
    sum_observations = np.sum(observation, axis=1)
    
    #Division factor for intensity
    sum_observations = sum_observations / obs_history.shape[1]

    fourier = fft(sum_observations)
    # Plot the result (the spectrum |Xk|)
    plt.figure(figsize=(9,3))
    plt.ylim(0, 10)
    plt.plot(np.abs(fourier))
    plt.title(f"FFT Sum_observations Episode {episode_id}")
    plt.show()

    """Plot the fft of ER-MRL agents context"""

    obs_RES_neurons = 100
    context = context_history[ep_step_start:ep_step_stop, :obs_RES_neurons]

    sum_contexts = np.sum(context, axis=1)

    sum_contexts = sum_contexts / context_history.shape[1]

    fourier = fft(sum_contexts)
    # Plot the result (the spectrum |Xk|)
    plt.figure(figsize=(9,3))
    plt.ylim(0, 10)
    plt.plot(np.abs(fourier))
    plt.title(f"FFT Sum_Context Episode {episode_id}")
    plt.show()

    ep_step_start += (ep_step_stop - ep_step_start)


def plot_observations_context(dones, obs_history, obs_idx_names, context_history, episode_id, ep_timesteps=None, legend=True, figsize=(16, 8)):
    ep_step_start = dones[episode_id - 1] if episode_id > 1 else 0 
    ep_step_stop = dones[episode_id]

    if ep_timesteps:
        nb_steps = ep_timesteps
    else:
        nb_steps = ep_step_stop - ep_step_start

    timesteps = np.arange(stop=nb_steps)

    fig, axes = plt.subplots(2, 1, figsize=figsize)

    """ Observation """
    ax_obs = axes[0]
    ax_obs.set_title(f"Observations Episode {episode_id}")

    for idx in range(len(obs_idx_names)):
        if ep_timesteps:
            observation = obs_history[ep_step_start:ep_step_start+ep_timesteps, idx]
        else:
            observation = obs_history[ep_step_start:ep_step_stop, idx]
        ax_obs.plot(timesteps, observation, label=obs_idx_names[idx])
        if legend:
            ax_obs.legend()

    """ RES_Context"""
    ax_ctx = axes[1]
    ax_ctx.set_title(f"RES_Context Episode {episode_id}")

    obs_RES_neurons = 20
    for idx in range(obs_RES_neurons):
        if ep_timesteps:
            context = context_history[ep_step_start:ep_step_start+ep_timesteps, idx]
        else:
            context = context_history[ep_step_start:ep_step_stop, idx]
        ax_ctx.plot(timesteps, context)

    plt.tight_layout()
    plt.savefig(f"figures/combined_plots_ep_{episode_id}")
    plt.show()
    plt.close()  # Close the figure to avoid displaying it immediately

    ep_step_start += (ep_step_stop - ep_step_start)


def get_random_seed():
    if os.environ.get('SLURM_ARRAY_TASK_ID'):
        seed = int(os.environ.get('SLURM_ARRAY_TASK_ID'))
    else:
        seed = 0

    return seed 

def get_models(del_obs):
    if del_obs == "True":
        lrs = [0.0003, 0.0003 , 0.0001, 0.00005]  
        models = ['PPO', 'RES_PPO_0.0003', 'RES_PPO_0.0001', 'RES_PPO_0.00005']
    elif del_obs == "False":
        lrs = [0.0003, 0.0003, 0.0001]  
        models = ['PPO', 'RES_PPO_0.0003', 'RES_PPO_0.0001']
    else:
        raise(ValueError("Unknown type of envs"))
    
    return lrs, models