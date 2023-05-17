CUDA_VISIBLE_DEVICES=0
torchrun --nnodes=1 --nproc_per_node=1 --master_port=25001 \
    llava/train/train_mem.py \
    --model_name_or_path /home/gpuall/hehx/PretrainedModels/LanguageModels/ChatModels/Chinese-Vicuna-lora-7b-belle-and-guanaco \
    --base_model_path /home/gpuall/hehx/PretrainedModels/LanguageModels/FoundationModels/llama-7b-hf \
    --data_path /home/gpuall/hehx/MLLM/data/wukong/chat.json \
    --image_folder /home/gpuall/hehx/MLLM/data/wukong/images \
    --vision_tower /home/gpuall/hehx/PretrainedModels/MultiModalModels/chinese-clip-vit-large-patch14 \
    --freeze_backbone True \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end \
    --bf16 True \
    --output_dir ./checkpoints/llava-7b-zh-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
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
    --lazy_preprocess True
    # --report_to wandb