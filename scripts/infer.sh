#!/bin/bash
python3 run_llava_zh.py \
--model-name $GEMINI_PREDATA/checkpoints/llava-13b-zh-pretrain \
--image-file images/llava_logo.png \
--query "请简要描述这张图片"
