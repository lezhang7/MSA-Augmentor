#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --partition=long                           # Ask for unkillable job
#SBATCH --cpus-per-task=4                              # Ask for 2 CPUs
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100l:1
#SBATCH --ntasks-per-node=1                               # Ask for 1 GPU
#SBATCH --mem=128G                                        # Ask for 10 GB of RAM
#SBATCH --time=20:00:00                                   

module load miniconda/3
conda init
conda activate openflamingo

cd ~/scratch/github_clone/MSA-Augmentor

python inference.py \
   --checkpoints ./checkpoints/msat5-base/checkpoint-740000/ \
   --data_path ./dataset/casp15/cfdb \
   --do_predict \
   --mode orphan \
   --a 1 \
   --t 10\
   --repetition_penalty 1.0 \
