# export PATH="/root/miniconda3/bin:$PATH"
# source activate
# conda activate
export CUDA_VISIBLE_DEVICES=1

torchrun --nnodes=1 --nproc_per_node=1 --master_port=25001 \
    llava/train/train_mem.py \
    --model_name_or_path /home/gpuall/hehx/PretrainedModels/LanguageModels/ChatModels/chinese-alpaca-plus-7b-hf \
    --data_path /home/gpuall/hehx/MLLM/data/caption/ai_challenger/caption_chat.json \
    --image_folder /home/gpuall/hehx/MLLM/data/caption/ai_challenger/images/ \
    --version 1 \
    --vision_tower /home/gpuall/hehx/PretrainedModels/MultiModalModels/chinese-clip-vit-large-patch14/ \
    --freeze_backbone True \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end \
    --bf16 False \
    --output_dir checkpoints/llava-13b-zh-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 8 \
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
