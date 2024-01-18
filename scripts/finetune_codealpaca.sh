MODEL_SIZE=7B
NUM_GPUS=8
BATCH_SIZE_PER_GPU=8
TOTAL_BATCH_SIZE=1024
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

export HF_DATASETS_CACHE="/group/40005/chengyuewu/hf_datasets_cache/"
export TRANSFORMERS_CACHE="/group/40005/chengyuewu/transformers_cache_dir/"

export NO_PROXY=localhost,.woa.com,.oa.com,.tencent.com,tencentcos.cn,myqcloud.com
export HTTP_PROXY=$ENV_VENUS_PROXY
export HTTPS_PROXY=$ENV_VENUS_PROXY
export no_proxy=$NO_PROXY
export http_proxy=$ENV_VENUS_PROXY
export https_proxy=$ENV_VENUS_PROXY

deepspeed open_instruct/finetune_trainer.py \
    --deepspeed ds_configs/stage1_no_offloading.conf \
    --model_name_or_path TencentARC/LLaMA-Pro-8B \
    --tokenizer_name TencentARC/LLaMA-Pro-8B \
    --use_fast_tokenizer False \
    --train_file data/processed/evol_codealpaca/evol_codealpaca_v1.jsonl \
    --max_seq_length 4096 \
    --do_train \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --evaluation_strategy "no" \
    --logging_steps 1 \
    --save_strategy epoch \
    --save_total_limit 3 \
    --num_train_epochs 3 \
    --preprocessing_num_workers 16 \
    --use_flash_attn \
    --use_checkpointing \
    --output_dir output/LLaMA-Pro-8B-evol-codealpaca \
    --bf16 \
    --tf32 True \
    --overwrite_output_dir \
    --report_to "none" 
