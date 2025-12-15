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

MODEL_PATH2=/lustre/projects/med-multi-llm/haohan/EasyR1/checkpoints/rif_pathvqa_clevr/qwen2_5_7b_pathvqa_clevr_random_1/global_step_8/actor/huggingface
MODEL_PATH3=/lustre/projects/med-multi-llm/haohan/EasyR1/checkpoints/rif_pathvqa_clevr/qwen2_5_7b_pathvqa_clevr_easy_1/global_step_8/actor/huggingface


python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=zhhxte/mllm_cl_pathvqa@train \
    data.val_files=zhhxte/mllm_cl_pathvqa@test \
    data.prompt_key=problem \
    data.image_key=images \
    worker.actor.model.model_path=${MODEL_PATH1} \
    worker.reward.reward_function=./examples/reward_function/math.py:compute_score \
    trainer.project_name=rif_pathvqa_clevr \
    trainer.experiment_name=rif_random_pathvqa_clevr_eval_pathvqa \
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
    worker.actor.model.model_path=${MODEL_PATH1} \
    worker.reward.reward_function=./examples/reward_function/math.py:compute_score \
    trainer.project_name=rif_pathvqa_clevr \
    trainer.experiment_name=rif_random_pathvqa_clevr_eval_mmmu \
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
    data.train_files=zhhxte/mllm_cl_pathvqa@train \
    data.val_files=zhhxte/mllm_cl_pathvqa@test \
    data.prompt_key=problem \
    data.image_key=images \
    worker.actor.model.model_path=${MODEL_PATH2} \
    worker.reward.reward_function=./examples/reward_function/math.py:compute_score \
    trainer.project_name=rif_pathvqa_clevr \
    trainer.experiment_name=rif_easy_pathvqa_clevr_eval_pathvqa \
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
    worker.actor.model.model_path=${MODEL_PATH2} \
    worker.reward.reward_function=./examples/reward_function/math.py:compute_score \
    trainer.project_name=rif_pathvqa_clevr \
    trainer.experiment_name=rif_easy_pathvqa_clevr_eval_mmmu \
    data.format_prompt=./examples/format_prompt/math.jinja \
    trainer.n_gpus_per_node=8 \
    data.image_key=images \
    worker.rollout.enforce_eager=false \
    data.filter_overlong_prompts=true \
    trainer.val_only=true \
    trainer.total_epochs=1
