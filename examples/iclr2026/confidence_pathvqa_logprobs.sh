#!/bin/bash
#SBATCH -o logs/%x/%j.out
#SBATCH -e logs/%x/%j.err
#SBATCH -p AISS2025073101
#SBATCH --nodes=1
#SBATCH --mem-per-gpu=200G
#SBATCH --cpus-per-gpu=22
#SBATCH -t 30-00:00:00
#SBATCH -J confidence_rl
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8

# saved for debugging
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

set -x
export https_proxy="http://cair:coy_suffocate_petrified@klb-fwproxy-01.aisc.local:3128"
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE

source .venv/bin/activate
MODEL_PATH=/lustre/projects/med-multi-llm/haohan/models/Qwen2.5-VL-7B-MedSFT

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=CAIR-HKISI/pathvqa_close_ended@train \
    data.val_files=CAIR-HKISI/pathvqa_close_ended@test \
    data.prompt_key=problem \
    data.image_key=images \
    data.shuffle=true \
    data.rollout_batch_size=128 \
    data.mini_rollout_batch_size=256 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.reward.reward_function=./examples/reward_function/no_cot_confidence_logprobs.py:compute_score \
    trainer.project_name=confidence_rl \
    trainer.experiment_name=grpo_shuffle_logbrobs_confidence_cares \
    data.format_prompt=./examples/format_prompt/cares.jinja \
    trainer.n_gpus_per_node=8 \
    trainer.save_model_only=true \
    data.image_key=images \
    worker.rollout.enforce_eager=false \
    data.filter_overlong_prompts=true \
    trainer.total_epochs=1

