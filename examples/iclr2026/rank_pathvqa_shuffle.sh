#!/bin/bash
#SBATCH -o logs/%x/%j.out
#SBATCH -e logs/%x/%j.err
#SBATCH -p AISS2025073101
#SBATCH --nodes=1
#SBATCH --mem-per-gpu=200G
#SBATCH --cpus-per-gpu=16
#SBATCH -t 30-00:00:00
#SBATCH -J rank_pathvqa
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8

# saved for debugging
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

set -x
export https_proxy="http://cair:coy_suffocate_petrified@klb-fwproxy-01.aisc.local:3128"
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE

source .venv/bin/activate
MODEL_PATH=/cm/shared/llm_models/Qwen/Qwen2.5-VL-7B-Instruct

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/lustre/projects/med-multi-llm/haohan/tpami/pathvqa_easyfirst/mllm_cl_pathvqa_rollout/train.parquet \
    data.val_files=/lustre/projects/med-multi-llm/haohan/tpami/pathvqa_easyfirst/mllm_cl_pathvqa_rollout/test.parquet \
    data.prompt_key=problem \
    data.image_key=images \
    data.shuffle=true \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.reward.reward_function=./examples/reward_function/math.py:compute_score \
    trainer.project_name=rank_pathvqa_shuffle \
    trainer.experiment_name=qwen2_5_7b_geo_pathvqa_easy \
    data.format_prompt=./examples/format_prompt/math.jinja \
    trainer.n_gpus_per_node=8 \
    trainer.save_model_only=true \
    data.image_key=images \
    worker.rollout.enforce_eager=false \
    data.filter_overlong_prompts=true \
    trainer.total_epochs=3
