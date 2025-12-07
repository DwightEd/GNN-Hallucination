import torch
from transformers import AutoTokenizer
import json

# # ======================
# # 加载数据
# # ======================
# data_PATH = "attention_hidden_data/HaluEval/test_layer24_QA/sample_0.pt"
response_path = "../../datasets/RAGTruth/response.jsonl"

# att_data = torch.load(response_path)

# print(att_data)
# model_name = "../../models/llama-2-7b-chat-hf"
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # RAGTruth 读取
response = []
with open(response_path, 'r') as f:
    for line in f:
        response.append(json.loads(line))

# num=0
for i in range(len(response)):
    if response[i]["source_id"]=="11412" and response[i]["model"]=="mistral-7B-instruct":

        print(response[i])
        break
#         print(f"幻觉段落：{response[i]['labels'][0]['text']},长度：{len(response[i]['labels'][0]['text'])}")
#         s_id=int(response[i]["labels"][0]["start"])
#         e_id=int(response[i]["labels"][0]["end"])

#         print(f"原文:{response[i]['response'][s_id:e_id+1]},长度：{len(response[i]['response'][s_id:e_id+1])}")
#         print(f"原文:{response[i]['response'][s_id:e_id]},长度：{len(response[i]['response'][s_id:e_id])}")
#         num+=1
#         print("========")
#     if num==5:
#         break
