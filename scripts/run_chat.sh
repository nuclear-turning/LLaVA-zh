#!/bin/bash

python serve/controller.py --host 0.0.0.0 --port 10000

python serve/model_worker.py --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path /home/gpuall/hehx/models/llava-zh-13b-finetune --multi-modal

python serve/gradio_web_server.py --controller http://localhost:10000