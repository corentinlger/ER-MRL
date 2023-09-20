import numpy as np
import matplotlib.pyplot as plt 

import gymnasium as gym
from gymnasium import spaces 

import reservoirpy as rpy
from reservoirpy.nodes import Reservoir


# TODO : Change the reservoir wrapper so only need 


class ReservoirWrapper(gym.Wrapper):
    """
    Implements a custom Reservoir Wrapper following the gym interface.At each time step, the observation, action, and reward of the agent
    are fed into a reservoir, which is a fixed recurrent neural network initialized with random weights. The reservoir is used to encode a 
    context that captures temporal dependencies between past observations, actions, and rewards.
    
    We use the term 'context' to refer to the state of the Reservoir neurons, which encode past information about the agent's interactions 
    with the environment (including the observation, action and reward associated). To avoid confusion, we return the context instead of the
    actual observation of the wrapped environment in the reset and step methods.
    
    :param env: (gym.Env) Gym environment that will be wrapped
    :param units (int) : The number of neurons in the Reservoir.
    :param lr (float) : The leak-rate of the Reservoir neurons
    :param sr (float) : The spectral radius of the Reservoir
    :param iss (float) : The input scaling of the Reservoir
    :param reset (bool) : Reset the Reservoir at each env reset or not
    :param skip_c (bool) : Do askip connection of the current (o, a, r) with the Reservoir context
    :param seed (int) : The random seed of the Reservoir
    """
    
    def __init__(self, env, units, lr, sr, iss, reset_res=True, skip_c=False, seed=42):
        super(ReservoirWrapper, self).__init__(env)
        
        self.env = env 
        
        
        self.reset_res = reset_res 
        self.skip_c = skip_c
        
        # if action_space is Discrete we need to one hot encode it to pass it to the Reservoir 
        self.discrete_a_space = isinstance(env.action_space, gym.spaces.Discrete)
        
        # Creation of a Reservoir with the given parameters 
        self.reservoir = Reservoir(units=units, lr=lr, sr=sr, input_scaling=iss, seed=seed)
        
        # Modifying the observation space to match the Reservoir neurons shape and dtype
        if self.skip_c : 
            if isinstance(env.action_space, spaces.Discrete):
                old_obs_space_features = env.observation_space.shape[0]
                old_act_space_features = env.action_space.n
                min_obs = min(env.observation_space.low)
                max_obs = min(env.observation_space.high)

            elif isinstance(env.action_space, spaces.Box):
                old_obs_space_features = env.observation_space.shape[0]
                old_act_space_features = env.action_space.shape[0]

                min_obs = np.min((min(env.observation_space.low), min(env.action_space.low)))
                max_obs = np.max((max(env.observation_space.high), max(env.action_space.high)))
            self.observation_space = spaces.Box(low=min_obs, high=max_obs, shape=(units + old_obs_space_features + old_act_space_features + 1,), dtype=np.float64)
        else:
            self.observation_space = spaces.Box(low=-1, high=1, shape=(units,), dtype=np.float64)
        
    
    def reset(self, seed=None, options=None):
        """
        Reset the wrapped environment and returns its new observation. Then, create two numpy arrays of zeros with action and reward shape
        and feed the Reservoir with a vector containing these observation, action and the reward to provide a new Reservoir Context. We choose
        to use arrays of zeros in order to give a neutral information every time an episode is reseted.
        
        We also give this information with a vector of shape (1, features) with features=concat(obs, action, reward) because the information
        given to a Reservoir must be of shape (nb_timesteps, nb_features in ReservoirPy).
        
        :return: (np.array, dict)
        """     
        # Reseting the Reservoir 
        if self.reset_res:
            self.reservoir.reset()

        # Getting the variables of self.env.reset
        obs, info = self.env.reset(seed=seed, options=options)
        obs = obs.reshape(1, -1)
        reward = np.zeros(1).reshape(1, -1)
        
        # Action encoding 
        action = self.one_hot_encode(0) if self.discrete_a_space else np.zeros(self.env.action_space.shape).reshape(1,-1)
    
        # Feeding the Reservoir with this data
        reservoir_input = np.concatenate((obs, action, reward), axis=1)
        context = self.reservoir(reservoir_input)
        
        # Add skip connection with (obs, action, reward)
        if self.skip_c:
            context = np.concatenate((context, obs, action, reward), axis=1)
        
        return context.flatten(), info 
    
    def step(self, action):
        """
        Make a step by applying the action chosen to the wrapped environment. The context the state of the Reservoir neurons after its activation
        by the (obs, action, reward) vector.
        
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, bool, dict) context, reward, is the episode over?, truncated, additional informations
        """
        
        # Getting the data from self.env.step
        obs, reward, done, truncated, info = self.env.step(action)
        obs = obs.reshape(1, -1)
        reward_obs = np.array(reward).reshape(-1, 1)
        
        # Action encoding 
        action = self.one_hot_encode(0) if self.discrete_a_space else np.zeros(self.env.action_space.shape).reshape(1,-1)
        
        # Feeding the Reservoir
        reservoir_input = np.concatenate((obs, action, reward_obs), axis=1)
        context = self.reservoir(reservoir_input)
        
        # Add skip connection with (obs, action, reward)
        if self.skip_c:
            context = np.concatenate((context, obs, action, reward_obs), axis=1)
        
        return context.flatten(), reward, done, truncated, info
    
    def one_hot_encode(self, action):
        """
        one hot encodes a Discrete action
        
        :param action: (int) Action taken by the agent
        :return: (np.ndarray) One hot encoded action
        """
        one_hot_vector = np.zeros(self.env.action_space.n).reshape(1,-1)
        one_hot_vector[0, action] = 1
        
        return one_hot_vector
    


class POReservoirWrapper(gym.Wrapper):
    """
    Implements a custom Reservoir Wrapper following the gym interface.At each time step, the observation, action, and reward of the agent
    are fed into a reservoir, which is a fixed recurrent neural network initialized with random weights. The reservoir is used to encode a 
    context that captures temporal dependencies between past observations, actions, and rewards.
    
    We use the term 'context' to refer to the state of the Reservoir neurons, which encode past information about the agent's interactions 
    with the environment (including the observation, action and reward associated). To avoid confusion, we return the context instead of the
    actual observation of the wrapped environment in the reset and step methods.
    
    :param env: (gym.Env) Gym environment that will be wrapped
    :param units (int) : The number of neurons in the Reservoir.
    :param lr (float) : The leak-rate of the Reservoir neurons
    :param sr (float) : The spectral radius of the Reservoir
    :param iss (float) : The input scaling of the Reservoir
    :param reset (bool) : Reset the Reservoir at each env reset or not
    :param skip_c (bool) : Do askip connection of the current (o, a, r) with the Reservoir context
    :param seed (int) : The random seed of the Reservoir
    """
    
    def __init__(self, env, units, obs_units, lr, sr, iss, reset_res=True, seed=42):
        super(POReservoirWrapper, self).__init__(env)
        
        self.env = env 
        
        self.obs_units = obs_units
        self.reset_res = reset_res 
        
        # if action_space is Discrete we need to one hot encode it to pass it to the Reservoir 
        self.discrete_a_space = isinstance(env.action_space, gym.spaces.Discrete)
        
        # Creation of a Reservoir with the given parameters 
        self.reservoir = Reservoir(units=units, lr=lr, sr=sr, input_scaling=iss, seed=seed)
        
        # Modifying the observation space to match the Reservoir neurons shape and dtype
        self.observation_space = spaces.Box(low=-1, high=1, shape=(obs_units,), dtype=np.float64)
        
    
    def reset(self, seed=None, options=None):
        """
        Reset the wrapped environment and returns its new observation. Then, create two numpy arrays of zeros with action and reward shape
        and feed the Reservoir with a vector containing these observation, action and the reward to provide a new Reservoir Context. We choose
        to use arrays of zeros in order to give a neutral information every time an episode is reseted.
        
        We also give this information with a vector of shape (1, features) with features=concat(obs, action, reward) because the information
        given to a Reservoir must be of shape (nb_timesteps, nb_features in ReservoirPy).
        
        :return: (np.array, dict)
        """     
        # Reseting the Reservoir 
        if self.reset_res:
            self.reservoir.reset()

        # Getting the variables of self.env.reset
        obs, info = self.env.reset(seed=seed, options=options)
        obs = obs.reshape(1, -1)
        reward = np.zeros(1).reshape(1, -1)
        
        # Action encoding 
        action = self.one_hot_encode(0) if self.discrete_a_space else np.zeros(self.env.action_space.shape).reshape(1,-1)
    
        # Feeding the Reservoir with this data
        reservoir_input = np.concatenate((obs, action, reward), axis=1)
        context = self.reservoir(reservoir_input)
        
        context = context.flatten()[:self.obs_units]
        
        return context, info 
    
    def step(self, action):
        """
        Make a step by applying the action chosen to the wrapped environment. The context the state of the Reservoir neurons after its activation
        by the (obs, action, reward) vector.
        
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, bool, dict) context, reward, is the episode over?, truncated, additional informations
        """
        
        # Getting the data from self.env.step
        obs, reward, done, truncated, info = self.env.step(action)
        obs = obs.reshape(1, -1)
        reward_obs = np.array(reward).reshape(-1, 1)
        
        # Action encoding 
        action = self.one_hot_encode(0) if self.discrete_a_space else np.zeros(self.env.action_space.shape).reshape(1,-1)
        
        # Feeding the Reservoir
        reservoir_input = np.concatenate((obs, action, reward_obs), axis=1)
        context = self.reservoir(reservoir_input)    
        
        context = context.flatten()[:self.obs_units]
        return context, reward, done, truncated, info
    
    def one_hot_encode(self, action):
        """
        one hot encodes a Discrete action
        
        :param action: (int) Action taken by the agent
        :return: (np.ndarray) One hot encoded action
        """
        one_hot_vector = np.zeros(self.env.action_space.n).reshape(1,-1)
        one_hot_vector[0, action] = 1
        
        return one_hot_vector
    

    


class MultiReservoirWrapper(gym.Wrapper):
    """
    
    !!! Description to do 
    Implements a custom  Reservoir Wrapper following the gym interface.At each time step, the observation, action, and reward of the agent
    are fed into a reservoir, which is a fixed recurrent neural network initialized with random weights. The reservoir is used to encode a 
    context that captures temporal dependencies between past observations, actions, and rewards.
    
    We use the term 'context' to refer to the state of the Reservoir neurons, which encode past information about the agent's interactions 
    with the environment (including the observation, action and reward associated). To avoid confusion, we return the context instead of the
    actual observation of the wrapped environment in the reset and step methods.
    
    :param env: (gym.Env) Gym environment that will be wrapped
    :param units (int) : The number of neurons in the Reservoir.
    :param lr (float) : The leak-rate of the Reservoir neurons
    :param sr (float) : The spectral radius of the Reservoir
    :param iss (float) : The input scaling of the Reservoir
    :param reset (bool) : Reset the Reservoir at each env reset or not
    :param skip_c (bool) : Do askip connection of the current (o, a, r) with the Reservoir context
    :param seed (int) : The random seed of the Reservoir
    """
    

    # plutôt faire qqch avec kwargs ? 
    def __init__(self, env, nb_res, reset_res=True, seed=42, **kwargs):
        super(MultiReservoirWrapper, self).__init__(env)
        
        self.env = env 
        
        self.nb_res = nb_res
        self.reset_res = reset_res 
        
        # check if action space is discrete or not 
        self.discrete_a_space = isinstance(env.action_space, gym.spaces.Discrete)

        # creation of Reservoirs with the given parameters 
        self.reservoirs = []
        parameters = ['units', 'iss', 'sr', 'lr']
        
        for res_id in range(nb_res):
            units, iss, sr, lr = [kwargs[f"{param}_{res_id+1}"] for param in parameters]
            reservoir = Reservoir(units=units, lr=lr, sr=sr, input_scaling=iss, seed=seed+res_id)
            self.reservoirs.append(reservoir)
            
            
        # Modifying the observation space to match the Reservoir neurons shape and dtype
        total_units = np.sum([kwargs[f"units_{res_idx+1}"] for res_idx in range(self.nb_res)])
        
        self.observation_space = spaces.Box(low=-1, high=1, shape=(total_units,), dtype=np.float64)
        
        
    
    def reset(self, seed=None, options=None):
        """
        Reset the wrapped environment and returns its new observation. Then, create two numpy arrays of zeros with action and reward shape
        and feed the Reservoir with a vector containing these observation, action and the reward to provide a new Reservoir Context. We choose
        to use arrays of zeros in order to give a neutral information every time an episode is reseted.
        
        We also give this information with a vector of shape (1, features) with features=concat(obs, action, reward) because the information
        given to a Reservoir must be of shape (nb_timesteps, nb_features in ReservoirPy).
        
        :return: (np.array, dict)
        """     

        # Reseting the Reservoir 
        if self.reset_res:
            for reservoir in self.reservoirs:
                reservoir.reset()

        # Getting the variables of self.env.reset
        obs, info = self.env.reset(seed=seed, options=options)
        obs = obs.reshape(1, -1)
        reward = np.zeros(1).reshape(1, -1)
        
        action = self.one_hot_encode(0) if self.discrete_a_space else np.zeros(self.env.action_space.shape).reshape(1,-1)
    
        reservoir_input = np.hstack((obs, action, reward))
        context = np.array([self.reservoirs[res_idx](reservoir_input) for res_idx in range(self.nb_res)]).reshape(1,-1)
    
        
        return context.flatten(), info 
    
    def step(self, action):
        """
        Make a step by applying the action chosen to the wrapped environment. The context the state of the Reservoir neurons after its activation
        by the (obs, action, reward) vector.
        
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, bool, dict) context, reward, is the episode over?, truncated, additional informations
        """
        
        # Getting the data from self.env.step
        obs, reward, done, truncated, info = self.env.step(action)
        obs = obs.reshape(1, -1)
        reward_obs = np.array(reward).reshape(1, -1)
        
        # Action encoding 
        action = self.one_hot_encode(0) if self.discrete_a_space else np.zeros(self.env.action_space.shape).reshape(1,-1)
        
        # Feeding the Reservoir
        reservoir_input = np.hstack((obs, action, reward_obs))
        context = np.array([self.reservoirs[res_idx](reservoir_input) for res_idx in range(self.nb_res)]).reshape(1,-1)

        return context.flatten(), reward, done, truncated, info
    
    def one_hot_encode(self, action):
        """
        one hot encodes a Discrete action
        
        :param action: (int) Action taken by the agent
        :return: (np.ndarray) One hot encoded action
        """
        one_hot_vector = np.zeros(self.env.action_space.n).reshape(1,-1)
        one_hot_vector[0, action] = 1
        
        return one_hot_vector
 
    
    
class RewardsSavingWrapper(gym.Wrapper):
        """
        This wrapper class permits to save the rewards obtained in an environment during N episodes. It will be used to evaluate the performance
        of the Reservoir + PPO model with specific Reservoir Hyperparameters (lr, sr, iss). 

        :param env (gym.Env): The gym environment wrapped
        """
        def __init__(self, env):
            super(RewardsSavingWrapper, self).__init__(env)

            self.env = env
            self.ep_rewards_list = []
            self.ep_reward = []

        def reset(self, seed=None, options=None):
            # if ep_reward non null
            if self.ep_reward: self.ep_rewards_list.append(np.sum(self.ep_reward))

            # reset the environment and the ep_reward
            obs, info= self.env.reset(seed=seed, options=options)
            self.ep_reward = []

            return obs, info

        def step(self, action):
            # do a normal step in the env and add the reward to the ep_reward
            context, reward, done, truncated, info = self.env.step(action)
            self.ep_reward.append(reward)

            return context, reward, done, truncated, info

        def mean_reward(self, last_n_ep):
            """
            Returns the mean reward on the last chosen number of episodes

            :param last_n_ep (int): Last n episodes on which mean reward is calculated
            :return mean_reward (float) : Mean reward on the last n_ep
            """
            rewards_last_np_ep = self.ep_rewards_list[-last_n_ep:]
            mean_reward = np.mean(rewards_last_np_ep)
            return mean_reward    
        
      
    
    
class DeletedVelocityWrapper(gym.ObservationWrapper):
    """
    Gym environment observation wrapper used to delete velocity terms in
    observations. The intention is the make the MDP partially observable.
    Adapted from MaskedVelocityWrapper of sb3_contrib.

    :param env: Gym environment
    """

    # Supported envs
    velocity_indices = {
        "CartPole-v1": np.array([1, 3]),
        "MountainCar-v0": np.array([1]),
        "MountainCarContinuous-v0": np.array([1]),
        "Pendulum-v1": np.array([2]),
        "LunarLander-v2": np.array([2, 3, 5]),
        "LunarLanderContinuous-v2": np.array([2, 3, 5]),
    }
    def __init__(self, env: gym.Env):
            super().__init__(env)

            env_id: str = env.unwrapped.spec.id
            try:
                # Delete velocity observations
                obs_shape = env.observation_space.shape[0]
                obs_indices = np.ones(obs_shape, dtype=bool)
                obs_indices[self.velocity_indices[env_id]] = False
                
                self.observation_space = gym.spaces.Box(
                    low=env.observation_space.low[obs_indices],
                    high=env.observation_space.high[obs_indices],
                    shape=(np.sum(obs_indices),),
                    dtype=np.float32,
                )

                self.observation_mask = obs_indices
            except KeyError as e:
                raise NotImplementedError(f"Velocity masking not implemented for {env_id}") from e
            
    def observation(self, observation: np.ndarray) -> np.ndarray:
        return observation[self.observation_mask]




class CollectObs_Wrapper(gym.Wrapper):
    
    def __init__(self, env):
        super(CollectObs_Wrapper, self).__init__(env)
        
        self.env = env
        self.observations_history = []
        self.actions_history = []
        self.dones_history = []
        self.truncated_history = []
        
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.observations_history.append(obs)
        self.actions_history.append(action)
        self.dones_history.append(done)
        self.truncated_history.append(truncated)
        return obs, reward, done, truncated, info 
               
               
class CollectContext_Wrapper(gym.Wrapper):
    
    def __init__(self, env):
        super(CollectContext_Wrapper, self).__init__(env)
        
        self.env = env
        self.contexts_history = []
        
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.contexts_history.append(obs)
        return obs, reward, done, truncated, info 
               
