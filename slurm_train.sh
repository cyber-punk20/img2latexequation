#!/bin/bash
#SBATCH --job-name=my_distributed_job
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:3
#SBATCH --distribution=cyclic:block
#SBATCH --partition=gpu
#SBATCH --time=01:00:00

# Your other SLURM options

# Make the train.sh script executable
chmod +x train.sh

# Run the distributed training using ibrun
mpiexec.hydra -hostfile ./hostfile -np 3 ./train.sh