# Using a Remote Cluster for Parallelized ER-MRL

1. **Transfer Code to the Remote Cluster**

```bash
rsync -av ER-MRL/ remote_cluster_address
```

2. **Evolve Reservoirs in Parallel**

- Use the provided [slurm file](parallelized_evolve_res.slurm) for efficient parallelization during the evolution phase. To run it, use this command line and specify the evolution environment :

```bash
sbatch parallelized_evolve_res.slurm HalfCheetah-v4
```

In this example, we launch the evolution phase on the HalfCheetah environment. The other experiment parameters (experiment name, number of training timesteps ...) are already encoded in the slurm file, so you can easiely modify them from there, or add them as arguments of the file and specify them in command line. 

3. **Test ER-MRL Agents in Parallel**

- Employ this [slurm file](parallelized_test.slurm) to test ER-MRL agents with evolved reservoirs in parallel. Use this command line to execute it : 

```bash
sbatch parallelized_test.slurm HalfCheetah-v4
```

4. **Retrieve Experiment Data**

Use the following scripts to synchronize data:
- [Synchronize Hyperparameters Evolution](syn_remote_optuna_storage.sh)
- [Synchronize ER-MRL Agents' Performance during testing phase](sync_remote_logs.sh)


## Additional Information

- For detailed usage instructions with more complex evolutions setups (multiple reservoirs, generalization experiments ...), refer to the [main readme](../README.md).


