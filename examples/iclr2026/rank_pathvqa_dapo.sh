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
#SBATCH --exclusive

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
    data.rollout_batch_size=128 \
    data.mini_rollout_batch_size=256 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.reward.reward_function=./examples/reward_function/dapo_zhh.py:compute_score \
    worker.reward.reward_function_kwargs='{"max_response_length":256,"overlong_buffer_length":64,"overlong_penalty_factor":1.0}' \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.28 \
    worker.actor.clip_ratio_dual=10.0 \
    worker.rollout.n=8 \
    worker.rollout.max_num_batched_tokens=16384 \
    worker.rollout.val_override_config='{"n":16,"temperature":1.0,"top_p":0.7}' \
    trainer.project_name=rank_pathvqa \
    trainer.experiment_name=dapo \
    data.format_prompt=./examples/format_prompt/math.jinja \
    trainer.n_gpus_per_node=8 \
    trainer.save_model_only=true \
    algorithm.disable_kl=True \
    algorithm.online_filtering=True \
    algorithm.filter_key=accuracy_normalized \
    algorithm.filter_low=0.01 \
    algorithm.filter_high=1.1 \
    data.image_key=images \
    trainer.max_try_make_batch=10 \
    worker.rollout.enforce_eager=false \
    data.filter_overlong_prompts=true \
    trainer.total_epochs=1
