from random import random
import pandas as pd
import json
def read_csv(csv_path):
    data = pd.read_csv(csv_path)
    return data.values.tolist()


import requests
import os
headers = {
    "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36"
}

prompts = [
    "请简要描述一下提供的图片。\n<image>",
    "精简地描述这张图片。\n<image>",
    "为给定的图片提供简短的描述。\n<image>",
    "为呈现的图片提供简洁的解释。\n<image>",
    "总结图片的视觉内容。\n<image>",
    "对接下来的图片给出简短明了的解释。\n<image>",
    "分享对提供的图片的简洁解读。\n<image>",
    "提供图片关键特征的简洁描述。\n<image>",
    "传达对展示的图片的简洁明了的描述。\n<image>",
    "提供照片的清晰且简洁的总结。\n<image>",
    "编写一份简洁但信息丰富的图片总结。\n<image>",
    "创建一份代表给出图像的紧凑叙述。\n<image>",
    
    "<image>\n请简要描述一下提供的图片。",
    "<image>\n精简地描述这张图片。",
    "<image>\n为给定的图片提供简短的描述。",
    "<image>\n为呈现的图片提供简洁的解释。",
    "<image>\n总结图片的视觉内容。",
    "<image>\n对上面的图片给出简短明了的解释。",
    "<image>\n分享对提供的图片的简洁解读。",
    "<image>\n提供图片关键特征的简洁描述。",
    "<image>\n传达对展示的图片的简洁明了的描述。",
    "<image>\n提供照片的清晰且简洁的总结。",
    "<image>\n编写一份简洁但信息丰富的图片总结。",
    "<image>\n创建一份代表给出图像的紧凑叙述。",
]

def down_(url,image_path):
    if not os.path.exists(image_path):
        res = requests.get(url,headers=headers)
        if res.status_code == 200:
            with open(image_path,'wb') as imb:
                imb.write(res.content)
            return 1
        else:
            return 0
    return 1
    

def down_images(data,image_dir):
    idx = 0
    
    for row in data:
        image_path = image_dir+'/'+str(idx)+'.jpg'
        down_(row[0],image_path)
            
        idx+=1
def check_images(data,image_dir):
    from PIL import Image
    idx = 0
    for row in data:
        image_path = image_dir+'/'+str(idx)+'.jpg'
        
        image = Image.open(image_path).convert('RGB')
        if image.size[0]<10 and image.size[1]<10:
            os.remove(image_path)
        idx+=1

def convert_(data,image_dir):
    result = []
    idx = 0
    for row in data:
        template_json = {
            "id":"",
            "image": "",
            "conversations": [
            {
                "from": "human",
                "value": random.choice(prompts)
            },
            {
                "from": "gpt",
                "value": ""
            }
            ]
        }
        image_path = image_dir+'/'+str(idx)+'.jpg'
        
        if os.path.exists(image_path):
            template_json["id"] = idx
            template_json["image"] = str(idx)+'.jpg'
            template_json["conversations"][-1]["value"] = row[1]
                
            result.append(template_json)
        idx +=1
    return result

def write2json(data):
    with open('/home/gpuall/hehx/MLLM/data/wukong/chat.json','w+') as jf:
        json.dump(data,jf,ensure_ascii=False)


    
data = read_csv('/home/gpuall/hehx/MLLM/data/wukong_release/wukong_100m_0.csv')
down_images(data,"/home/gpuall/hehx/MLLM/data/wukong_new/images")
# result = convert_(data,"/home/gpuall/hehx/MLLM/data/wukong/images")
# write2json(result)
