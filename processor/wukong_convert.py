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
                "value": "请简要描述一下提供的图片.\n<image>"
            },
            {
                "from": "gpt",
                "value": ""
            }
            ]
        }
        image_path = image_dir+'/'+str(idx)+'.jpg'
        if down_(row[0],image_path):
            template_json["id"] = idx
            template_json["image"] = str(idx)+'.jpg'
            template_json["conversations"][-1]["value"] = row[1]
            idx +=1
            result.append(template_json)
    return result

def write2json(data):
    with open('/home/gpuall/hehx/MLLM/data/wukong/chat.json','w+') as jf:
        json.dump(data,jf,ensure_ascii=False)


    
data = read_csv('/home/gpuall/hehx/MLLM/data/wukong_release/wukong_100m_0.csv')[:40000]
result = convert_(data,"/home/gpuall/hehx/MLLM/data/wukong/images")
write2json(result)