#!/bin/bash
#SBATCH --job-name=visa-test # create a short name for your job
#SBATCH --output=visa-test-out.txt
#SBATCH --error=visa-test-error.txt
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=4     # total number of tasks across all nodes
#SBATCH --cpus-per-task=32 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --time=202:00:00          # total run time limit (HH:MM:SS)
source /apps/local/conda_init.sh
conda activate video-proj
export OMP_NUM_THREADS=64
# bash ./scripts_smit/latent_code_single.sh
bash ./scripts_smit/latent_code_mm.sh