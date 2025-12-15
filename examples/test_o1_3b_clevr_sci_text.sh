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

MODEL_PATH1=/lustre/projects/med-multi-llm/haohan/models/Qwen2.5-VL-3B-Instruct
MODEL_PATH2=/lustre/projects/med-multi-llm/haohan/EasyR1/checkpoints/easy_r1/qwen2_5_3b_nocot_sci_02/global_step_36/actor/huggingface
MODEL_PATH3=/lustre/projects/med-multi-llm/haohan/EasyR1/checkpoints/easy_r1/qwen2_5_3b_nocot_clevr_sci_text_03/global_step_67/actor/huggingface


python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=zhhxte/mllm_cl_scienceqa@train \
    data.val_files=zhhxte/mllm_cl_scienceqa@test \
    data.prompt_key=problem \
    data.image_key=images \
    worker.actor.model.model_path=${MODEL_PATH1} \
    worker.reward.reward_function=./examples/reward_function/rft.py:no_cot \
    trainer.experiment_name=qwen2_5_3b_eval_sci \
    data.format_prompt=./examples/format_prompt/no_cot.jinja \
    trainer.n_gpus_per_node=8 \
    data.image_key=images \
    worker.rollout.enforce_eager=false \
    data.filter_overlong_prompts=true \
    trainer.val_only=true \
    trainer.total_epochs=1

sleep 15

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=zhhxte/mllm_cl_textvqa@train \
    data.val_files=zhhxte/mllm_cl_textvqa@test \
    data.prompt_key=problem \
    data.image_key=images \
    worker.actor.model.model_path=${MODEL_PATH1} \
    worker.reward.reward_function=./examples/reward_function/rft.py:no_cot \
    trainer.experiment_name=qwen2_5_3b_eval_text \
    data.format_prompt=./examples/format_prompt/no_cot.jinja \
    trainer.n_gpus_per_node=8 \
    data.image_key=images \
    worker.rollout.enforce_eager=false \
    data.filter_overlong_prompts=true \
    trainer.val_only=true \
    trainer.total_epochs=1
    
sleep 15    

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=zhhxte/mllm_cl_clevr@train \
    data.val_files=zhhxte/mllm_cl_clevr@test \
    data.prompt_key=problem \
    data.image_key=images \
    worker.actor.model.model_path=${MODEL_PATH2} \
    worker.reward.reward_function=./examples/reward_function/rft.py:no_cot \
    trainer.experiment_name=qwen2_5_3b_clevr_sci_eval_clevr \
    data.format_prompt=./examples/format_prompt/no_cot.jinja \
    trainer.n_gpus_per_node=8 \
    data.image_key=image \
    worker.rollout.enforce_eager=false \
    data.filter_overlong_prompts=true \
    trainer.val_only=true \
    trainer.total_epochs=1

sleep 15    
   
python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=zhhxte/mllm_cl_clevr@train \
    data.val_files=zhhxte/mllm_cl_clevr@test \
    data.prompt_key=problem \
    data.image_key=images \
    worker.actor.model.model_path=${MODEL_PATH3} \
    worker.reward.reward_function=./examples/reward_function/rft.py:no_cot \
    trainer.experiment_name=qwen2_5_3b_clevr_sci_text_eval_clevr \
    data.format_prompt=./examples/format_prompt/no_cot.jinja \
    trainer.n_gpus_per_node=8 \
    data.image_key=image \
    worker.rollout.enforce_eager=false \
    data.filter_overlong_prompts=true \
    trainer.val_only=true \
    trainer.total_epochs=1
    
sleep 15
    
python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=zhhxte/mllm_cl_scienceqa@train \
    data.val_files=zhhxte/mllm_cl_scienceqa@test \
    data.prompt_key=problem \
    data.image_key=images \
    worker.actor.model.model_path=${MODEL_PATH3} \
    worker.reward.reward_function=./examples/reward_function/rft.py:no_cot \
    trainer.experiment_name=qwen2_5_3b_clevr_sci_text_eval_sci \
    data.format_prompt=./examples/format_prompt/no_cot.jinja \
    trainer.n_gpus_per_node=8 \
    data.image_key=images \
    worker.rollout.enforce_eager=false \
    data.filter_overlong_prompts=true \
    trainer.val_only=true \
    trainer.total_epochs=1
