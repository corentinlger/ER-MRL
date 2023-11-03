# Using a Remote Cluster for Parallelized ER-MRL

1. **Transfer Code to the Remote Cluster**

```bash
rsync -av ER-MRL/ remote_cluster_address
```

2. **Evolve Reservoirs in Parallel**

- Use the provided [slurm file](parallelized_evolve_res.slurm) for efficient parallelization during the evolution phase.


3. **Test ER-MRL Agents in Parallel**

- Employ this [slurm file](parallelized_test.slurm) to test ER-MRL agents with evolved reservoirs in parallel.

4. **Retrieve Experiment Data**

Use the following scripts to synchronize data:
- [Synchronize Hyperparameters Evolution](syn_remote_optuna_storage.sh)
- [Synchronize ER-MRL Agents' Performance](sync_remote_logs.sh)


## Additional Information

- For detailed usage instructions with more complex evolutions (multiple reservoirs, generalization experiments ...), refer to the [main readme](../README.md).


