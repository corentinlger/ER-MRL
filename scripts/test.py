import os
import time
import datetime
import argparse

import gymnasium as gym
import optuna
from optuna.storages import JournalStorage, JournalFileStorage
from stable_baselines3 import PPO

from er_mrl.wrappers import ReservoirWrapper, DeletedVelocityWrapper
from er_mrl.utils import get_random_seed, get_models

if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, required=True)
    parser.add_argument('--h_test', type=str, required=True)
    parser.add_argument('--units', type=int, required=False, default=100)
    parser.add_argument('--del_obs', type=str, required=False, default='False')
    parser.add_argument('--fixed_iss', type=str, required=False, default='True')
    parser.add_argument('--HP_training_steps', type=int, required=False, default=300000)
    parser.add_argument('--training_steps', type=int, required=False, default=10000000)
    parser.add_argument('--skip_c', type=str, required=False, default='False')

    # Choose the random seed for testing
    args = parser.parse_args()

    # If you are using slurm to parallelize tests, get the slurm task id as random seed
    seed = get_random_seed()

    # Create the experiment name of the evolution phase
    HP_exp_name = f"{args.env_id}_{args.h_test}_{args.HP_training_steps//1000}k_steps"
    log_name = f"optuna-journal_{HP_exp_name}.log"
    storage = JournalStorage(JournalFileStorage(log_name))

    # Load the results of the evolution phase
    study = optuna.load_study(
        study_name = f'{HP_exp_name}',
        storage = storage
    )

    # Getting best hyperparameters from evolution phase
    units = args.units
    lr = study.best_params['lr']
    sr = study.best_params['sr']
    iss = study.best_params['iss'] if args.fixed_iss == 'False' else 1.0
    skip_c = True if args.skip_c == 'True' else False

    # Create directories to store the results of the testing phase
    exp_name = f"{args.env_id}_{args.h_test}"
    logdir = os.path.join('logs', exp_name)
    models_dir = os.path.join('models', exp_name)

    os.makedirs(logdir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Test the ER-MRL agents with different learning rates 
    lrs, models = get_models(args.del_obs)

    # Run the experiment 
    start = time.time()
    for i, model_name in enumerate(models):
            lrn_r = lrs[i]
            env = gym.make(args.env_id)

            if args.del_obs == 'True':
                env = DeletedVelocityWrapper(env)
            if model_name.startswith('RES'):
                env = ReservoirWrapper(env, seed=seed, units=args.units, lr=lr, sr=sr, iss=iss, skip_c=skip_c)

            model = PPO('MlpPolicy', env, verbose=0, learning_rate=lrn_r, tensorboard_log=logdir, seed=seed)
            model.learn(total_timesteps=args.training_steps, tb_log_name=f"{model_name}_{int(args.training_steps/1000)}k_steps_seed_{seed}")
            model.save(f"{models_dir}/{model_name}_{int(args.training_steps/1000)}k_steps_seed_{seed}")
    end = time.time()

    print(f"Testing done in {str(datetime.timedelta(seconds=end-start))}")