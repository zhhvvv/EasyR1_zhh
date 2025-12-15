#!/bin/bash
#SBATCH -o logs/%x/%j.out
#SBATCH -e logs/%x/%j.err
#SBATCH -p AISS2025073101
#SBATCH --nodes=1
#SBATCH --mem-per-gpu=200G
#SBATCH --cpus-per-gpu=16
#SBATCH -t 30-00:00:00
#SBATCH -J qwen3_o1_mmmu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8

# saved for debugging
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

set -x
export https_proxy="http://cair:coy_suffocate_petrified@klb-fwproxy-01.aisc.local:3128"
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE

source .venv/bin/activate

MODEL_PATH1=/lustre/projects/med-multi-llm/haohan/models/Qwen3-VL-8B-Instruct
MODEL_PATH2=/lustre/projects/med-multi-llm/haohan/EasyR1/checkpoints/easy_r1/qwen3_nocot_clevr_sci_02/global_step_36/actor/huggingface
MODEL_PATH3=/lustre/projects/med-multi-llm/haohan/EasyR1/checkpoints/easy_r1/qwen3_8b_nocot_clevr_sci_text_03/global_step_67/actor/huggingface


python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=zhhxte/mllm_cl_mmmu@test \
    data.val_files=zhhxte/mllm_cl_mmmu@test \
    data.prompt_key=problem \
    data.image_key=images \
    worker.actor.model.model_path=${MODEL_PATH1} \
    worker.reward.reward_function=./examples/reward_function/math.py:compute_score \
    trainer.project_name=eval_mmmu \
    trainer.experiment_name=qwen3_8b_eval_mmmu_cot \
    data.format_prompt=./examples/format_prompt/math.jinja \
    trainer.n_gpus_per_node=8 \
    data.image_key=images \
    worker.rollout.enforce_eager=false \
    data.filter_overlong_prompts=true \
    trainer.val_only=true \
    trainer.total_epochs=1
    
sleep 15
    
python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=zhhxte/mllm_cl_mmmu@test \
    data.val_files=zhhxte/mllm_cl_mmmu@test \
    data.prompt_key=problem \
    data.image_key=images \
    worker.actor.model.model_path=${MODEL_PATH3} \
    worker.reward.reward_function=./examples/reward_function/math.py:compute_score \
    trainer.project_name=eval_mmmu_cot \
    trainer.experiment_name=qwen3_8b_clevr_sci_text_eval_mmmu \
    data.format_prompt=./examples/format_prompt/math.jinja \
    trainer.n_gpus_per_node=8 \
    data.image_key=images \
    worker.rollout.enforce_eager=false \
    data.filter_overlong_prompts=true \
    trainer.val_only=true \
    trainer.total_epochs=1
