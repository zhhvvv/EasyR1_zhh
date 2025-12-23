#!/bin/bash
#SBATCH -o logs/%x/%j.out
#SBATCH -e logs/%x/%j.err
#SBATCH -p AISS2025073101
#SBATCH --nodes=1
#SBATCH --mem-per-gpu=200G
#SBATCH --cpus-per-gpu=16
#SBATCH -t 30-00:00:00
#SBATCH -J confidence_rl
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive

# saved for debugging
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

wandb login

set -x
export https_proxy="http://cair:coy_suffocate_petrified@klb-fwproxy-01.aisc.local:3128"
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE

source .venv/bin/activate
MODEL_PATH=CAIR-HKISI/Qwen2.5-VL-7B-MedSFT

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/lustre/projects/med-multi-llm/haohan/tpami/pathvqa_easyfirst/mllm_cl_pathvqa_rollout/train.parquet \
    data.val_files=/lustre/projects/med-multi-llm/haohan/tpami/pathvqa_easyfirst/mllm_cl_pathvqa_rollout/test.parquet \
    data.prompt_key=problem \
    data.image_key=images \
    data.shuffle=true \
    data.rollout_batch_size=128 \
    data.mini_rollout_batch_size=256 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.logprobs=5 \
    worker.reward.reward_function=./examples/reward_function/nocot_with_confidence_token.py:compute_score \
    trainer.project_name=confidence_rl \
    trainer.experiment_name=grpo_shuffle_nocot_with_confidence_token \
    data.format_prompt=./examples/format_prompt/cares.jinja \
    trainer.n_gpus_per_node=8 \
    trainer.save_model_only=true \
    data.image_key=images \
    worker.rollout.enforce_eager=false \
    data.filter_overlong_prompts=true \
    trainer.total_epochs=3

