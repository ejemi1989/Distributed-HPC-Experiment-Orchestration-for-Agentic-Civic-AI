#!/bin/bash
#SBATCH --job-name=agentic-ai
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

module load python
source venv/bin/activate

ray start --head --port=6379 &
sleep 10

ray start --address=$SLURM_NODELIST:6379 &

python src/distributed_train.py
