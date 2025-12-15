#!/bin/bash
#SBATCH -o logs/%x/%j.out
#SBATCH -e logs/%x/%j.err
#SBATCH -p AISS2025073101
#SBATCH --nodes=1
#SBATCH --mem-per-gpu=200G
#SBATCH --cpus-per-gpu=16
#SBATCH -t 30-00:00:00
#SBATCH -J RFTvsSFT
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8

# saved for debugging
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

set -x
export https_proxy="http://cair:coy_suffocate_petrified@klb-fwproxy-01.aisc.local:3128"
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE

source .venv/bin/activate
MODEL_PATH=/lustre/projects/med-multi-llm/haohan/models/Qwen3-VL-8B-Instruct  # replace it with your local file path
#MODEL_PATH=/cm/shared/llm_models/Qwen/Qwen2.5-VL-7B-Instruct
#MODEL_PATH=/cm/shared/llm_models/meta-llama/Llama-3.2-11B-Vision-Instruct

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=zhhxte/mllm_cl_clevr@train \
    data.val_files=zhhxte/mllm_cl_clevr@test \
    data.prompt_key=problem \
    data.image_key=image \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.reward.reward_function=./examples/reward_function/rft.py:no_cot \
    trainer.experiment_name=qwen3_nocot_clevr_01 \
    data.format_prompt=./examples/format_prompt/no_cot.jinja \
    trainer.n_gpus_per_node=8 \
    data.image_key=image \
    worker.rollout.enforce_eager=false \
    data.filter_overlong_prompts=true \
    trainer.total_epochs=3 

