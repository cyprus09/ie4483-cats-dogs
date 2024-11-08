#!/bin/bash
#SBATCH --job-name=alexnet_1   # Job name
#SBATCH --output=train_1.txt    # Output file (%j is replaced with job ID)
#SBATCH --error=error_%j.txt      # Error file (%j is replaced with job ID)
#SBATCH --ntasks-per-node=4                # Number of tasks (jobs)
#SBATCH --time=01:00:00           # Maximum runtime (1 hour)
#SBATCH --partition=Lab2080       # Partition to submit to
#SBATCH --nodes=3
#SBATCH --gpus-per-node=4
#SBATCH --time=021:00:00
#SBATCH --cpus-per-task=4

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=enp179s0f1
export NCCL_P2P_DISABLE=1

export MASTER_PORT=55000
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

# Commands to execute
source ~/miniconda3/bin/activate
conda activate ai_project
echo "Starting the job..."
srun python train.py

