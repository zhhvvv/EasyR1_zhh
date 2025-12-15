#!/bin/bash
#SBATCH -o logs/%x/%j.out
#SBATCH -e logs/%x/%j.err
#SBATCH -p AISS2025073101
#SBATCH --nodes=1
#SBATCH --mem-per-gpu=200G
#SBATCH --cpus-per-gpu=16
#SBATCH -t 30-00:00:00
#SBATCH -J roll_pathvaq_new
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=klb-dgx-009,klb-dgx-015

python examples/roll_pathvqa_new.py
