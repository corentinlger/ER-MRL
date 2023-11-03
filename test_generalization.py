import os 
import time
import datetime 
import argparse

import numpy as np
import gymnasium as gym 
import optuna
from optuna.storages import JournalStorage, JournalFileStorage
from stable_baselines3 import PPO

from ER_MRL.wrappers import MultiReservoirWrapper

parser = argparse.ArgumentParser()
parser.add_argument('--env_id', type=str, required=True)
parser.add_argument('--HP_env_type', type=str, required=True)
parser.add_argument('--h_test', type=str, required=True)
parser.add_argument('--nb_res', type=int, required=True)
parser.add_argument('--units', type=int, required=False, default=100)
parser.add_argument('--HP_training_steps', type=int, required=False, default=300000)
parser.add_argument('--training_steps', type=int, required=False, default=10000000) 

args = parser.parse_args()
if os.environ.get('SLURM_ARRAY_TASK_ID'):
    seed = int(os.environ.get('SLURM_ARRAY_TASK_ID'))
else:
    seed = 0

HP_exp_name = f"{args.HP_env_type}_{args.nb_res}_{args.h_test}_{args.HP_training_steps//1000}k_steps"
log_name = f"optuna-journal_{HP_exp_name}.log"
storage = JournalStorage(JournalFileStorage(log_name))

study = optuna.load_study(
    study_name = f'{HP_exp_name}',
    storage = storage
)  

params = study.best_params
for res_id in range(1, args.nb_res + 1):
    params[f"units_{res_id}"] = args.units

# New training Experiment
exp_name = f"{args.env_id}_{args.h_test}"
logdir = os.path.join('logs', exp_name)
models_dir = os.path.join('models', exp_name)

os.makedirs(logdir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

lrs = [0.0003, 0.0003]
models = ['PPO', 'RES_PPO_0.0003']

# Experiment run 
start = time.time()

for i, model_name in enumerate(models):
    lrn_r = lrs[i]
    env = gym.make(args.env_id)
    obs, info = env.reset(seed=seed)
    
    if model_name.startswith('RES'):
        env = MultiReservoirWrapper(env, nb_res=args.nb_res, seed=seed, **params)
        for reservoir in env.reservoirs:
            print(reservoir.get_param)
        
    model = PPO('MlpPolicy', env, verbose=0, learning_rate=lrn_r, tensorboard_log=logdir, seed=seed)
    model.learn(total_timesteps=args.training_steps, tb_log_name=f"{model_name}_{int(args.training_steps/1000)}k_steps_seed_{seed}")
    model.save(f"{models_dir}/{model_name}_{int(args.training_steps/1000)}k_steps_seed_{seed}")

end = time.time()
print(f"Training done in {str(datetime.timedelta(seconds=end-start))}")
