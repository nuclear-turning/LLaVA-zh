
from transformers import LlamaForCausalLM,AutoTokenizer
from deltalm.modeling_deltalm import DeltalmForConditionalGeneration
import torch
from tqdm import tqdm
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
# 值得注意的句子：
# The bus in the image is white and red.（红白相间）
# what could be the possible reasons for the man sitting on top of the possessions in the back of the pickup truck?
class E2CTranslator:
    def __init__(self,model_cls,model_path,tokenizer_path):
        self.model = model_cls.from_pretrained(model_path).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model.eval()

        # text = "what could be the possible reasons for the man sitting on top of the possessions in the back of the pickup truck?"


    def trans(self,texts):
        if not isinstance(texts,list):
            texts = [texts]
        with torch.no_grad():
            batch_input = self.tokenizer(texts,return_tensors="pt",padding="longest").to("cuda")
            batch_output = self.model.generate(**batch_input)#,max_new_tokens=1024)
            batch_result = self.tokenizer.batch_decode(batch_output,skip_special_tokens=True)
            torch.cuda.empty_cache()
            del batch_output,batch_input
        return batch_result

def read_json(data_path):
    with open(data_path) as jf:
        data = json.load(jf)
    return data
def write_json(data,output_path):
    with open(output_path,'w+') as jf:
        json.dump(data,jf,ensure_ascii=False,indent=4)

def check_and_clean():
    pass
import copy
def do_trans(e2c_tool,data):
    result = []
    for item in tqdm(data):
        try: 
            temp_data = []
            temp_file = "temp/"+item['id']+'.json'
            if os.path.exists(temp_file):
                continue
            item['lang'] = "en"
            temp_data.append(copy.deepcopy(item))
            convers_text = []
            for convers in item["conversations"]:
                raw_text = convers['value'].replace("\n<image>","").replace("<image>\n","")
                convers_text.append(raw_text)
            trans_result = e2c_tool.trans(convers_text)
            # print(trans_result)
            for idx,text in enumerate(trans_result):
                item["conversations"][idx]["value"] = item["conversations"][idx]["value"].replace(convers_text[idx],text)
                # print(idx,text,item,item["conversations"][idx])
            del trans_result
            item['lang'] = "zh"
            temp_data.append(item)
            write_json(temp_data,temp_file)
            # result.append(item)
        except Exception as e:
            print(e)
    return result


def main():
    randeng_path = "/home/gpuall/hehx/PretrainedModels/LanguageModels/SmallModels/Randeng-Deltalm-362M-En-Zh"
    randeng_tokenizer = "/home/gpuall/hehx/PretrainedModels/LanguageModels/SmallModels/infoxlm-base"
    ziya_path = "/home/gpuall/hehx/PretrainedModels/LanguageModels/FoundationModels/Ziya-LLaMA-13B-v1.1"
    
    json_path = "/home/gpuall/hehx/MLLM/data/vqa/llava/llava_instruct_150k.json"
    model_type = "randeng"
    trans_models = {
        "randeng":randeng_path,
        "ziya":ziya_path
        }
    model_cls = {
        "randeng":DeltalmForConditionalGeneration,
        "ziya":LlamaForCausalLM
    }
    tokenizer = {
        "randeng":randeng_tokenizer,
        "ziya":ziya_path
    }
    output_path = "/home/gpuall/hehx/MLLM/data/vqa/llava_zh/llava_instruct_150k_zh.json"
    
    e2c_tool = E2CTranslator(model_cls[model_type],trans_models[model_type],tokenizer[model_type])
    data = read_json(json_path)
    result = do_trans(e2c_tool,data)
    # write_json(data,output_path)

if __name__ == "__main__":
    main()
    