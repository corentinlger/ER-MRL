import time
import datetime
import argparse

import numpy as np
import gymnasium as gym
import optuna
from optuna.storages import JournalStorage, JournalFileStorage
from stable_baselines3 import PPO 

from ER_MRL.wrappers import MultiReservoirWrapper, RewardsSavingWrapper
from ER_MRL.experiments import generate_env_ids, env_target_reward, env_random_reward, generate_params, generate_sampler

optuna.logging.set_verbosity(optuna.logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument('--nb_trials', type=int, required=True)
parser.add_argument('--sampler', type=str, required=True)
parser.add_argument('--h_test', type=str, required=True)
parser.add_argument('--env_type', type=str, required=True)
parser.add_argument('--nb_res', type=int, required=True)
parser.add_argument('--training_steps', type=int, required=True)
parser.add_argument('--nb_seeds', type=int, required=False, default=3)
parser.add_argument('--ep_eval', type=int, required=False, default=10)
parser.add_argument('--units', type=int, required=False, default=100)
parser.add_argument('--skip_c', type=str, required=False, default=False)

parser.add_argument('--min_sr', type=float, required=False, default=1e-4)
parser.add_argument('--max_sr', type=float, required=False, default=1e2)
parser.add_argument('--min_lr', type=float, required=False, default=1e-2)
parser.add_argument('--max_lr', type=float, required=False, default=1)
parser.add_argument('--min_iss', type=float, required=False, default=1e-1)
parser.add_argument('--max_iss', type=float, required=False, default=10)

args = parser.parse_args()

# Create the name of the experiment to save the results in relevant directories
exp_name = f"{args.env_type}_{args.nb_res}_{args.h_test}_{args.training_steps//1000}k_steps"
print(f"exp name : {exp_name}\n")

# Create an optuna storage to save the hyperparameters optimization 
log_name = f"optuna-journal_{exp_name}.log"
storage = JournalStorage(JournalFileStorage(log_name))

def objective(trial):
    """
      Objective function for each trial (i.e. each iteration of the evolutionary loop)
      For each trial, reservoir hyperparameters are sampled from a distrubution, and 
      they are tested within ER-MRL agents with different random seeds on the chosen 
      set of environments
      """
    print(f"\nTrial number {trial.number}")
    normalized_rewards = []
    env_ids = generate_env_ids(args.env_type)
    params = generate_params(trial, args)

    # We test the hyperparameters on the different chosen environments
    for env_id in env_ids:
        env_rewards = []
        # We test the hyperparameters within agents with different random seeds 
        for seed in range(args.nb_seeds):
            seed_rewards = []
            env = gym.make(env_id)

            env = MultiReservoirWrapper(env, nb_res=args.nb_res, seed=seed, **params)
            env = RewardsSavingWrapper(env)
            
            model = PPO('MlpPolicy', env, verbose=0, seed=seed)
            model.learn(total_timesteps=args.training_steps)
            
            # We get agent performance (i.e. the mean reward on the last n episodes)
            mean_last_rewards = env.mean_reward(last_n_ep=args.ep_eval)
            env_rewards.append(mean_last_rewards)

        # At the end of training for an environment :
        mean_env_reward = np.mean(env_rewards)
        normalized_env_reward = (mean_env_reward - env_random_reward[env_id]) / (env_target_reward[env_id] - env_random_reward[env_id])
        normalized_rewards.append(normalized_env_reward)
        print(f"\nenv_id : {env_id}  ;  normalized rewards : {normalized_rewards}")
        
    # Do the mean of normalized rewards for all envs
    fitness_score = [np.mean(normalized_rewards)]
    print(f"{fitness_score = }")
    # Return the fitness for this set of hyperparameters  
    return fitness_score

# Create an Optuna sampler (i.e. the optimization algorithm in the outer loop)
sampler = generate_sampler(args.sampler)

# Create an Optuna study with the sampler and storage chosen 
study = optuna.create_study(
    study_name=f'{exp_name}',
    direction="maximize",
    sampler=sampler,
    storage=storage,
    load_if_exists=True)

# Start the study (i.e. launch the evolution phase)
start = time.time()
study.optimize(objective, n_trials=args.nb_trials)
end = time.time()

print(f"\nOptimization took {str(datetime.timedelta(seconds=end-start))} for {args.nb_trials} trials on {args.env_type} tasks")
