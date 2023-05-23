export CUDA_VISIBLE_DEVICES=3,5,6
torchrun --nnodes=1 --nproc_per_node=1 --master_port=25001 \
    llava/train/train_mem.py \
    --model_name_or_path /home/gpuall/hehx/PretrainedModels/LanguageModels/ChatModels/Chinese-Vicuna-7b \
    --data_path /home/gpuall/hehx/MLLM/data/vqa/CH-QA.json \
    --image_folder /home/gpuall/hehx/MLLM/data/vqa/images \
    --vision_tower /home/gpuall/hehx/PretrainedModels/MultiModalModels/chinese-clip-vit-large-patch14 \
    --freeze_backbone True \
    --pretrain_mm_mlp_adapter /home/gpuall/hehx/MLLM/LLaVA-zh_/checkpoints/mm_projector/llava-7b-zh-pretrain-v1.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --bf16 True \
    --output_dir ./checkpoints/llava-7b-zh-finetune \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
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
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb
    # --fsdp "full_shard auto_wrap" \
    # --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer'