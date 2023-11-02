# Evolving-Reservoirs-for-Meta-Reinforcement-Learning


Code for the `Evolving-Reservoirs-for-Meta-Reinforcement-Learning` paper. Our goal is to study the following question : How neural structures, optimized at an evolutionary scale, can enhance the capabilities of agents to learn complex tasks at a developmental scale?

Our repository provides:

### Python scripts to :
- **evolve ER-MRL agents with [one](evolve_res.py) or [multiple](evolve_multi_res.py) reservoir(s) on the same environment**
- **evolve ER-MRL agents [with multiple reservoir on different environments](evolve_generalization.py) to study generalization**
- **evaluate evolved ER-MRL agents against standard RL agents with the different setups mentioned above**

### Files to :
- **Use existing or custom [Gymnasium](https://gymnasium.farama.org/index.html) Reinforcement Learning environments**
- **[Mask observations from an environment](ER_MRL/wrappers.py) and study POMDP or [create your own reservoir architecture inside an ER-MRL agent](ER_MRL/wrappers.py) with gymnasium wrappers**

### A [tutorial to parallelized our method](parallelization_tutorials/) :
- **transfer the code on a CPU cluster**
- **run the evolution phase ER-MRL agents in parallel**
- **evaluate evolved ER-MRL agents against standard RL agents in parallel**
- **retrieve the data to analyze the experiments results locally**

 ### A [notebook](analyze_results.ipynb/) to show how to analyze experiments results

## Installation (Delete my name for the anonymous repository)

1- Get the repository

```bash
git clone https://github.com/anonymous/Evolving-Reservoirs-for-Meta-Reinforcement-Learning.git
cd Evolving-Reservoirs-for-Meta-Reinforcement-Learning/
```
2- Install the dependencies 

```bash
python -m venv myvenv
source myvenv/bin/activate
pip install -r requirements.txt
```

## Usage

### Evolve reservoirs and test ER-MRL agents on the same environment

To evolve and find the best reservoir structure within an ER-MRL agent on a specific task, you can use `evolve_res.py`. You will need to specify some parameters like in the following example (you can take a look at the complete list of argument in [this script](evolve_res.py)) : 

```bash
python evolve.py --env_id HalfCheetah-v4 --h_test test_experiment --training_steps 300000
```

We recommend using few training timesteps for the evolution part, because each RL training (dev scale) can be long ... (you can test final HPs on more timesteps). 
It is also recommended to use a parallelized approach to run the evolution experiments, for example by following this [notebook tutorial](url) to run evolutionary algorithm on parallelized jobs with a cluster of CPUs.

If you want to test your best agent, you can use the following command (make sure you provide the parameters corresponding to the ones used in the evolution phase):

```bash
python test.py --env_id HalfCheetah-v4 --h_test test_experiment --HP_training_steps 300000
```

TODO : say you can watch the complete analysis (HPs + test ) in this example jupyter notebook 
The results of the evolution of reservoir hyperparameters will be stored within an optuna storage. You can analyze your results with the following [notebook](url)

TODO : say the procedure is the same for multiple reservoirs, just specify the number of reservoirs ... and other params in (link file)

### Study generalization of neural structures by evolving and testing agents on different environments


To do this, go in [this file](ER_MRL/experiments.py) and add your own env_type to the ```generate_env_ids``` function as follows :  

```python
elif env_type == 'your_env_type':   
            env_ids = ['your_env_id_1', 'your_env_id_2', ...]
```

To run an experiment, first evolve the reservori(s) like this : 

```bash
python evolve_multi_envs.py --nb_res 2 --env_type Ant_Swimmer --h_test generalization_test_experiment
```

And then test the results like this : 

```bash
python3 test_generalization.py --nb_res 2 --HP_env_type Ant_Swimmer --env_id HalfCheetah-v4 --h_test generalization_test_experiment
```


## Development

- Clean and refactor the code. Add tutorials. Add more functionnalities

