import optuna

def generate_params(trial, args):
      # Generated HPs 
      hps = ['lr', 'sr']
      if args.fixed_iss == "False":
            hps.append('iss')

      params = {}
      for hp in hps:
            min_param = getattr(args, f"min_{hp}")
            max_param = getattr(args, f"max_{hp}")
            for res_id in range(1, args.nb_res+1):
                  params[f"{hp}_{res_id}"] = trial.suggest_float(f"{hp}_{res_id}", min_param, max_param, log=True)

      # Fixed HP (units)
      for res_id in range(1, args.nb_res+1):
            params[f"units_{res_id}"] = args.units
            if args.fixed_iss == "True":
                  params[f"iss_{res_id}"] = 1.0
                 
      return params

def generate_sampler(sampler):
      if sampler == 'Random':
            sampler = optuna.samplers.RandomSampler()
      elif sampler == 'Tpe':
            sampler = optuna.samplers.TPESampler()
      elif sampler =='Cmaes':
            sampler = optuna.samplers.CmaEsSampler(restart_strategy='ipop')
            print("restart_strategy ipop")
      return sampler

def generate_env_ids(env_type):
      if env_type == 'Ant_Swimmer':
            env_ids = ['Ant-v4','Swimmer-v4']

      elif env_type == 'Ant_Cheetah':
            env_ids = ['Ant-v4','HalfCheetah-v4']

      elif env_type == 'Cheetah_Swimmer':
            env_ids = ['HalfCheetah-v4','Swimmer-v4'
            ]

      elif env_type == 'Ant_4_tr_envs':
            env_ids = ['HalfCheetah-v4', 'Hopper-v4', 'Swimmer-v4', 'Walker2d-v4'
            ]

      elif env_type == 'Cheetah_4_tr_envs':
            env_ids = ['Ant-v4','Hopper-v4','Swimmer-v4', 'Walker2d-v4'
            ]

      elif env_type == 'Swimmer_4_tr_envs':
            env_ids = ['Ant-v4', 'HalfCheetah-v4', 'Hopper-v4', 'Walker2d-v4'
            ]

      elif env_type == 'Hopper_4_tr_envs':
            env_ids = ['Ant-v4', 'HalfCheetah-v4', 'Swimmer-v4', 'Walker2d-v4'
            ]

      elif env_type == 'Walker_4_tr_envs':
            env_ids = ['Ant-v4', 'HalfCheetah-v4', 'Swimmer-v4', 'Hopper-v4'
            ]
      
      elif env_type == 'forward_mujoco':   
            env_ids = ['Ant-v4', 'HalfCheetah-v4', 'Hopper-v4', 'Swimmer-v4', 'Walker2d-v4'
                       ]
      return env_ids

# Update values with PPO benchmark code
env_target_reward = {  'Ant-v4': 125,
                       'HalfCheetah-v4': 750,
                       'Swimmer-v4': 60,
                       'Hopper-v4': 1250, 
                       'Humanoid-v4': 375,
                       'HumanoidStandup-v4': 7500,
                       'Walker2d-v4': 800}

env_random_reward = {  'Ant-v4': -0.355 ,
                       'HalfCheetah-v4': -0.292,
                       'Hopper-v4': 0.751,
                       'Humanoid-v4': 5.025,
                       'HumanoidStandup-v4': 33.089,
                       'InvertedDoublePendulum-v4': 9.088,
                       'InvertedPendulum-v4': 1.0,
                       'Pusher-v4': -1.493,
                       'Reacher-v4': -0.849,
                       'Swimmer-v4': 0.0,
                       'Walker2d-v4': 0.075}