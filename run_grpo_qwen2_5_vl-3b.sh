set -x

# 统一的输出根目录
RUN_OUTPUT_DIR="/home/data1/gjl/verl_results/$(date +%Y%m%d_%H%M%S)_finaldata_test"
mkdir -p "$RUN_OUTPUT_DIR"

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=/home/data1/gjl/verl_v0.6.1
export SWANLAB_API_KEY=R0sBHF3RSabpNsyIwSyJA
export SWANLAB_LOG_DIR="$RUN_OUTPUT_DIR/swanlab_logs"

export HF_DATASETS_CACHE="/home/data1/gjl/.cache/hf/hf_datasets"
export RAY_TMPDIR="/home/data1/gjl/.cache/ray_tmp"
export TMPDIR="/home/data1/gjl/.cache/system_tmp"

# 禁用代理（避免连接问题）
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset all_proxy
unset ALL_PROXY

# 使用HuggingFace镜像站点（如果需要）
export HF_ENDPOINT=https://hf-mirror.com

ENGINE=${1:-vllm}

echo "=================================="
echo "运行输出目录: $RUN_OUTPUT_DIR"
echo "=================================="

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/home/data1/gjl/dataset/final_dataset/train.parquet \
    data.val_files=/home/data1/gjl/dataset/final_dataset/val.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=multi_modal_data \
    actor_rollout_ref.model.path=Qwen/Qwen3-VL-4B-Instruct \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","swanlab"]' \
    trainer.default_local_dir="/home/data1/gjl/checkpoints" \
    trainer.project_name='final_data' \
    trainer.log_val_generations=10 \
    trainer.validation_data_dir="$RUN_OUTPUT_DIR/val_dump" \
    +trainer.output_dir="$RUN_OUTPUT_DIR/outputs" \
    trainer.experiment_name='grpo-qwen3-vl-4b-instruct' \
    hydra.run.dir="$RUN_OUTPUT_DIR/hydra_outputs" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.resume_mode=disable \
    trainer.total_epochs=1 $@

echo "=================================="
echo "训练完成! 所有输出文件保存在:"
echo "$RUN_OUTPUT_DIR"
echo "=================================="
