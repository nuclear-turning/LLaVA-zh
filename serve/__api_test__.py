import requests
from conversation import (default_conversation, conv_templates,
                                   SeparatorStyle)
chat_api = "http://localhost:21002/chat-stream"

temperature=0.1
max_new_tokens=2048

def create_prompt(text, image, image_process_mode):
    assert image_process_mode in ["Crop", "Resize", "Pad"]
    text = text[:1536]  # Hard cut-off
    if image is not None:
        text = text[:1200]  # Hard cut-off for images
        if '<image>' not in text:
            text = text + '\n<image>'
        text = (text, image, image_process_mode)
    new_state = conv_templates["llava_v1"].copy()
    new_state.append_message(new_state.roles[0], text)
    new_state.append_message(new_state.roles[1], None)
    state = new_state
    # Construct prompt
    prompt = state.get_prompt()
    pload = {
        "model": "llava_zh",
        "prompt": prompt,
        "temperature": float(temperature),
        "max_new_tokens": min(int(max_new_tokens), 1536),
        "stop": state.sep if state.sep_style == SeparatorStyle.SINGLE else state.sep2,
        "images": f'List of {len(state.get_images())}',
    }
    pload['images'] = state.get_images()
    state.messages[-1][-1] = "▌"
    return pload

from PIL import Image
import json
image = Image.open("/home/gpuall/hehx/MLLM/LLaVA-zh/serve/examples/extreme_ironing.jpg")
pload = create_prompt("图片上的男子坐的车是黄色的吗",image,"Crop")
print(pload["prompt"])
res = requests.post(
    chat_api,
    json=pload,
    stream=True
)
def post_process_code(code):
    sep = "\n```"
    if sep in code:
        blocks = code.split(sep)
        if len(blocks) % 2 == 1:
            for i in range(1, len(blocks), 2):
                blocks[i] = blocks[i].replace("\\_", "_")
        code = sep.join(blocks)
    return code
for chunk in res.iter_lines(decode_unicode=False, delimiter=b"\0"):
    if chunk:
        data = json.loads(chunk.decode())
        if data["error_code"] == 0:
            output = data["text"].strip()
            output = post_process_code(output)
            print(output)