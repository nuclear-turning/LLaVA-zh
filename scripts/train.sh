# export GPU_NUM=2 TRAIN_BATCH=24 EVAL_BATCH=16
torchrun --nnodes=1 --nproc_per_node=$GPU_NUM --master_port=25001 \
    llava/train/train_mem.py \
    --model_name_or_path $GEMINI_DATA_IN1/Ziya-LLaMA-13B-v1.1 \
    --data_path $GEMINI_DATA_IN3/caption/ai_challenger/caption_chat.json \
    --image_folder $GEMINI_DATA_IN3/caption/ai_challenger/images/ \
    --version 1 \
    --vision_tower $GEMINI_DATA_IN2/chinese-clip-vit-large-patch14/ \
    --freeze_backbone True \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end \
    --bf16 False \
    --output_dir $GEMINI_DATA_OUT/llava-zh-13b-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size $TRAIN_BATCH \
    --per_device_eval_batch_size $EVAL_BATCH \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --tf32 False \
    --report_to tensorboard
