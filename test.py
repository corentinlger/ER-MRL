import os
import time
import datetime
import argparse
import numpy as np

from stable_baselines3 import PPO

import gymnasium as gym

import optuna
from optuna.storages import JournalStorage, JournalFileStorage

from Wrappers import ReservoirWrapper, DeletedVelocityWrapper

parser = argparse.ArgumentParser()


# General params
parser.add_argument('--env_id', type=str, required=True)
parser.add_argument('--h_test', type=str, required=True)
parser.add_argument('--units', type=int, required=True)
parser.add_argument('--del_obs', type=str, required=False, default='False')
parser.add_argument('--fixed_iss', type=str, required=False, default='True')
parser.add_argument('--HP_training_steps', type=int, required=False, default=300000)
parser.add_argument('--training_steps', type=int, required=False, default=10000000)
parser.add_argument('--skip_c', type=str, required=False, default='False')

# No need Reservoir params cuz use best ones in Optuna Storage

args = parser.parse_args()
seed = int(os.environ.get('SLURM_ARRAY_TASK_ID'))

# Getting best_params from HP Search
HP_exp_name = f"{args.env_id}_{args.h_test}_{args.HP_training_steps//1000}k_steps"
print(f"HP_exp name : {HP_exp_name}\n")
log_name = f"optuna-journal_{HP_exp_name}.log"
storage = JournalStorage(JournalFileStorage(log_name))

study = optuna.load_study(
    study_name = f'{HP_exp_name}',
    storage = storage
)

# best RES_params
units = args.units
lr = study.best_params['lr']
sr = study.best_params['sr']
iss = study.best_params['iss'] if args.fixed_iss == 'False' else 1.0
skip_c = True if args.skip_c == 'True' else False

print(f"{iss = }")
print(f"{skip_c = }")

# New training Experiment
exp_name = f"{args.env_id}_{args.h_test}"
logdir = os.path.join('logs', exp_name)
models_dir = os.path.join('models', exp_name)

os.makedirs(logdir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

if args.del_obs == "True":
    lrs = [0.0003, 0.0003 , 0.0001, 0.00005]  
    models = ['PPO', 'RES_PPO_0.0003', 'RES_PPO_0.0001', 'RES_PPO_0.00005']
elif args.del_obs == "False":
    lrs = [0.0003, 0.0003, 0.0001]  
    models = ['PPO', 'RES_PPO_0.0003', 'RES_PPO_0.0001']
else:
     raise(ValueError("Unknown type of envs"))

print(f"{lrs = }")
print(f"{models = }")


# Experiment run

start = time.time()

for i, model_name in enumerate(models):
        lrn_r = lrs[i]
        env = gym.make(args.env_id)

        if args.del_obs == 'True':
            env = DeletedVelocityWrapper(env)
            print("Velocity deleted")
        if model_name.startswith('RES'):
            env = ReservoirWrapper(env, seed=seed, units=args.units, lr=lr, sr=sr, iss=iss, skip_c=skip_c)
            print(env.reservoir.get_param)

        model = PPO('MlpPolicy', env, verbose=0, learning_rate=lrn_r, tensorboard_log=logdir, seed=seed)
        model.learn(total_timesteps=args.training_steps, tb_log_name=f"{model_name}_{int(args.training_steps/1000)}k_steps_seed_{seed}")
        model.save(f"{models_dir}/{model_name}_{int(args.training_steps/1000)}k_steps_seed_{seed}")

end = time.time()
print(f"Training done in {str(datetime.timedelta(seconds=end-start))}")