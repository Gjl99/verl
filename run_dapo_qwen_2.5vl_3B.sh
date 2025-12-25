#!/usr/bin/env bash
set -xeuo pipefail

# 禁用代理
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset all_proxy
unset ALL_PROXY

# 使用HuggingFace镜像站点
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 核心环境/路径设置 
# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"/home/data1/gjl/verl_v0.6.1"} 
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
# 路径 
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-VL-3B-Instruct"}
CKPTS_DIR=${CKPTS_DIR:-"${WORKING_DIR}/results/outputs"} 
TRAIN_FILE=${TRAIN_FILE:-"/home/data1/gjl/dataset/elecVerse_0915_grpo/split_output/train.parquet"}
TEST_FILE=${TEST_FILE:-"/home/data1/gjl/dataset/elecVerse_0915_grpo/split_output/val.parquet"}
CURRENT_DATE=$(date +%Y-%m-%d)
#actor_rollout_ref.actor.optim.name=adam_offload \
ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
    --working-dir "${WORKING_DIR}" \
    -- python3 -m recipe.dapo.main_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.image_key=multi_modal_data \
    data.truncation='left' \
    data.filter_overlong_prompts=False \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.gen_batch_size=12 \
    data.train_batch_size=4 \
    actor_rollout_ref.rollout.n=16 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.max_num_gen_batches=0 \
    algorithm.filter_groups.metric=acc \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=3072 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=3072 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=3072 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=50 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode="token-mean" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.60 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=3072 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k="-1" \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=True \
    reward_model.overlong_buffer.len=512 \
    reward_model.overlong_buffer.penalty_factor=1.0 \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name="verl_elecVerse_dapo_3B" \
    trainer.experiment_name="DAPO-Qwen2.5VL-3B" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.test_freq=20 \
    trainer.save_freq=100 \
    trainer.total_epochs=3 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    +trainer.output_dir=/home/data1/gjl/verl_old/verl/results/outputs \
    hydra.run.dir="/home/data1/gjl/verl_old/verl/results/outputs/${CURRENT_DATE}/DAPO-Qwen2.5VL-3B"