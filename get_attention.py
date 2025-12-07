import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def calculate_hallucination_spans(labels, text, response_rag, tokenizer, end_inclusive=False):
    """
    labels: list of dicts with 'start' and 'end' (char-level relative to response_rag)
    text: prompt after tokenizer.apply_chat_template(...) (string)
    response_rag: generated response (string)
    tokenizer: huggingface tokenizer
    end_inclusive: if True, assume label['end'] is inclusive char index; 
                   if False, assume exclusive (i.e., Python slice style)
    returns: list of [token_start_in_input, token_end_in_input_exclusive]
             -- token_end_in_input_exclusive uses half-open interval, suitable for range(start, end)
    """

    # tokenize response alone to get offsets
    rag_enc = tokenizer(
        response_rag,
        return_offsets_mapping=True,
        add_special_tokens=False
    )
    offsets = rag_enc["offset_mapping"]  # list of (char_s, char_e) pairs, char_e is exclusive

    # prefix length: tokenize the prompt/text *without* adding special tokens
    prefix_enc = tokenizer(text, add_special_tokens=False)
    prefix_len = len(prefix_enc["input_ids"])

    spans = []
    for item in labels:
        char_s = item["start"]
        char_e = item["end"]
        # normalize: treat char_e as exclusive if end_inclusive=False
        if end_inclusive:
            char_e = char_e + 1  # convert to exclusive

        # find token indices that cover [char_s, char_e)
        token_s = None
        token_e = None  # exclusive index

        for idx, (s, e) in enumerate(offsets):
            # token covers chars [s, e) (e exclusive)
            if token_s is None and s <= char_s < e:
                token_s = idx
            # we want token_e such that offsets[token_e-1] covers up to char_e-1
            if token_s is not None and s < char_e <= e:
                token_e = idx + 1  # exclusive
                break

        # fallback: if char_s maps to token boundary but char_e on next token, try widen
        if token_s is not None and token_e is None:
            # find last token whose end <= char_e, include up to that token
            for j in range(token_s, len(offsets)):
                s, e = offsets[j]
                if e >= char_e:
                    token_e = j + 1
                    break
            if token_e is None:
                token_e = len(offsets)

        if token_s is None:
            # label starts before first token? skip or set to 0
            # We'll skip adding this span (could log)
            continue

        # convert to input-level token indices by adding prefix_len
        spans.append([prefix_len + token_s, prefix_len + token_e])  # end exclusive

    return spans


def get_token_hallucination_labels(seq_len, hallucination_spans):
    """
    spans: list of [start, end_exclusive] token indices
    returns labels length seq_len of 0/1
    """
    labels = [0] * seq_len
    for s, e in hallucination_spans:
        # ensure bounds
        s = max(0, s)
        e = min(seq_len, e)
        for i in range(s, e):
            labels[i] = 1
    return labels
model_name="../../models/Mistral-7B-Instruct-v0.3"
response_path = "../../datasets/RAGTruth/response.jsonl"
source_info_path="../../datasets/RAGTruth/source_info.jsonl"


#RAGtruth数据集读取
response = []
source_info_dict = {}
with open(response_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        response.append(data)

with open(source_info_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        source_info_dict[data['source_id']] = data

# print(response[0])
# first_key, first_value = next(iter(source_info_dict.items()))
# print(first_key, first_value)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = "cuda:0"
data_len=0

split_type="train"
select_layer=24  #24 28 32
task_type="Summary"

save_dir = f"attention_hidden_data/RAGtruth/Mistral-7B-Instruct-v0.3/{split_type}_layer{select_layer}_{task_type}"
import os
os.makedirs(save_dir, exist_ok=True)

for i in tqdm(range(len(response)),desc="Data processing"):
    if response[i]['model'] == "mistral-7B-instruct" and response[i]["split"] == split_type:  ##test数据总共450条，其中有幻觉的为226条
        response_rag = response[i]['response']
        source_id = response[i]['source_id']
        temperature = response[i]['temperature']
        if source_info_dict[source_id]['task_type']==task_type:#train 839条QA
            data_len+=1
            # if train_QA_len==3:
            #     print(i)
            #     break
            prompt =  source_info_dict[source_id]['prompt']  ##test max_prompt_len = 9973   train max_prompt_len = 10719
            messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ]
            
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
    
            input_text = text+response_rag
            
            input_ids = tokenizer([input_text], return_tensors="pt").input_ids.to(device)
            prefix_ids = tokenizer([text], return_tensors="pt").input_ids.to(device)
            
            if "labels" in response[i].keys():
                hallucination_spans = calculate_hallucination_spans(response[i]['labels'], text, response_rag, tokenizer)
            else:
                hallucination_spans = []
            
                # token-level 0/1 label
            hallucination_labels = get_token_hallucination_labels(
                seq_len=input_ids.shape[-1],
                hallucination_spans=hallucination_spans
            )
            # task_type = source_info_dict[source_id]['task_type']
            with torch.no_grad():
                outputs = model(input_ids=input_ids,
                        return_dict=True,
                        output_attentions=True,
                        output_hidden_states=True
                        )
            # hidden_states = outputs.hidden_states[select_layer]
            # attention = outputs.attentions[select_layer-1]  
            all_attentions = torch.stack(
                            [att.squeeze(0).cpu() for att in outputs.attentions], 
                            dim=0
                        )
            response_idx = prefix_ids.shape[-1]
            # V=[]
            # for ids in input_ids.squeeze(0).cpu():
            #     decoded_text = tokenizer.decode([ids], skip_special_tokens=False)
            #     V.append(decoded_text)
                
            save_obj={
                "source_id": source_id,
                "response_idx":response_idx,
                "token_ids": input_ids.squeeze(0).cpu(),         # (seq,)
                # "hidden_states": hidden_states.squeeze(0).cpu(),            # (seq, dim)
                "attention": all_attentions,                    # (layers,heads, seq, seq)
                "hallucination_labels": hallucination_labels, 
                "original_idx":i,# list[int], len = seq
                # "V":V,
            }
            save_path = f"{save_dir}/sample_{data_len}.pt"
            torch.save(save_obj, save_path)
            torch.cuda.empty_cache()
print(f"保存{data_len}条数据到{save_dir}")
# 最终保存
# torch.save(saved_results, f"attention_hidden_data/saved_{split_type}_layer{select_layer}_with_{task_type}_num{data_len}.pt")
# print(f"保存完成：attention_hidden_data/saved_{split_type}_layer{select_layer}_with_{task_type}_num{data_len}.pt")