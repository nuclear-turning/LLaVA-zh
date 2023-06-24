export PATH="/root/miniconda3/bin:$PATH"
source activate
conda activate
torchrun --nnodes=1 --nproc_per_node=$GPU_NUM --master_port=25001 \
    llava/train/train_mem.py \
    --model_name_or_path $GEMINI_DATA_IN1/Chinese-alpaca-13b-plus \
    --version 1 \
    --data_path $GEMINI_DATA_IN2/llava_zh_instruct/llava_instruct_150k_zh.json \
    --image_folder $GEMINI_DATA_IN2/llava_zh_instruct/images \
    --vision_tower $GEMINI_DATA_IN1/chinese-clip-vit-large-patch14 \
    --freeze_backbone False \
    --pretrain_mm_mlp_adapter $GEMINI_DATA_IN1/llava-zh-mm-projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --bf16 True \
    --output_dir $GEMINI_DATA_OUT/checkpoints/llava-zh-13b-finetune \
    --num_train_epochs 3 \
    --per_device_train_batch_size $TRAIN_BATCH \
    --per_device_eval_batch_size $EVAL_BATCH \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb
    # --fsdp "full_shard auto_wrap" \
    # --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer'
