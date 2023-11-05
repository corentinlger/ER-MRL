import time
import datetime
import argparse

import numpy as np
import gymnasium as gym
import optuna
from optuna.storages import JournalStorage, JournalFileStorage
from stable_baselines3 import PPO 

from ER_MRL.wrappers import ReservoirWrapper, RewardsSavingWrapper, DeletedVelocityWrapper
from ER_MRL.experiments import generate_sampler


optuna.logging.set_verbosity(optuna.logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument('--nb_trials', type=int, required=True)
parser.add_argument('--h_test', type=str, required=True)
parser.add_argument('--env_id', type=str, required=True)
parser.add_argument('--training_steps', type=int, required=True)
parser.add_argument('--sampler', type=str, required=True)
parser.add_argument('--del_obs', type=str, required=False, default='False')
parser.add_argument('--nb_seeds', type=int, required=False, default=3)
parser.add_argument('--ep_eval', type=int, required=False, default=10)
parser.add_argument('--units', type=int, required=False, default=100)
parser.add_argument('--skip_c', type=str, required=False, default=False)
parser.add_argument('--fixed_iss', type=str, required=False, default='True')

parser.add_argument('--min_sr', type=float, required=False, default=1e-4)
parser.add_argument('--max_sr', type=float, required=False, default=1e2)
parser.add_argument('--min_lr', type=float, required=False, default=1e-2)
parser.add_argument('--max_lr', type=float, required=False, default=1)
parser.add_argument('--min_iss', type=float, required=False, default=1e-1)
parser.add_argument('--max_iss', type=float, required=False, default=10)

args = parser.parse_args()

# Create the name of the experiment to save the results in relevant directories
exp_name = f"{args.env_id}_{args.h_test}_{args.training_steps//1000}k_steps"
print(f"exp name : {exp_name}\n")

# Create an optuna storage to save the hyperparameters optimization 
log_name = f"optuna-journal_{exp_name}.log"
storage = JournalStorage(JournalFileStorage(log_name))

def objective(trial):
    """
      Objective function for each trial (i.e. each iteration of the evolutionary loop)
      For each trial, reservoir hyperparameters are sampled from a distrubution, and 
      they are tested within ER-MRL agents with different random seeds on the chosen 
      environment
      """
    print(f"\nTrial number {trial.number}")
    rewards = []

    # Trial hyperparameters to optimize
    units = args.units
    if args.fixed_iss == "True":
      iss = 1.0
    else:
      iss = trial.suggest_float("iss", args.min_iss, args.max_iss, log=True)
    sr = trial.suggest_float("sr", args.min_sr, args.max_sr, log=True)
    lr = trial.suggest_float("lr", args.min_lr, args.max_lr, log=True)
    skip_c = True if args.skip_c == 'True' else False

    # We test the hyperparameters within agents with different random seeds 
    for seed in range(args.nb_seeds):
        # We launch a whole RL training loop with the chosen HP in the Reservoir 
        env = gym.make(args.env_id)
        if args.del_obs == "True":
             env = DeletedVelocityWrapper(env)
        env = ReservoirWrapper(env, seed=seed,units=units, lr=lr, sr=sr, iss=iss, skip_c=skip_c)
        env = RewardsSavingWrapper(env)
        
        model = PPO('MlpPolicy', env, verbose=0, seed=seed)
        model.learn(total_timesteps=args.training_steps)
        
        # We get agent performance (i.e. the mean reward on the last n episodes)
        mean_last_rewards = env.mean_reward(last_n_ep=args.ep_eval)
        rewards.append(mean_last_rewards)

    # Return the fitness for this set of hyperparameters  
    return np.mean(rewards)

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

print(f"Optimization took {str(datetime.timedelta(seconds=end-start))} for {args.nb_trials} trials on {args.env_id} task")
