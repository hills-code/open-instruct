MODEL_SIZE=8B
NUM_GPUS=8
BATCH_SIZE_PER_GPU=2
TOTAL_BATCH_SIZE=1024
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"


deepspeed open_instruct/finetune_trainer.py \
    --deepspeed $DS_CONFIG_PATH \
    --model_name_or_path $YOUR_EXPANDED_MODEL_PATH \
    --tokenizer_name $ORIGINAL_TOKENIZER_PATH \
    --use_fast_tokenizer False \
    --dataset_name HuggingFaceTB/cosmopedia  \
    --max_seq_length 8192 \
    --do_train \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 1e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.06 \
    --weight_decay 0. \
    --gradient_clipping 1.0 \
    --evaluation_strategy "no" \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 50 \
    --use_synthetic \
    --num_train_epochs 2 \
    --save_total_limit 3 \
    --output_dir $OUTPUT_PATH \
    --bf16 \
    --use_flash_attn \
    --tf32 True \
    --use_checkpointing \
    --report_to "none" \
    --preprocessing_num_workers 128 
