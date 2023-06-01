export PATH="/root/miniconda3/bin:$PATH"
source activate
conda activate
export CUDA_VISIBLE_DEVICES=0,1,2,3,
torchrun --nnodes=1 --nproc_per_node=2 --master_port=25001 \
    llava/train/train_mem.py \
    --model_name_or_path $GEMINI_DATA_IN1/Chinese-alpaca-13b-plus \
    --data_path $GEMINI_DATA_IN2/aic_wukong/chat_caption.json \
    --image_folder $GEMINI_DATA_IN2/aic_wukong/images/ \
    --version 1 \
    --vision_tower $GEMINI_DATA_IN1/chinese-clip-vit-large-patch14 \
    --freeze_backbone True \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end \
    --bf16 True \
    --output_dir $GEMINI_DATA_OUT/checkpoints/llava-13b-zh-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
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
    --tf32 True \
    --report_to tensorboard