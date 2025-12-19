import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
import os

model_name="../../models/Mistral-7B-Instruct-v0.3"
data_path = "../../datasets/Text-sql/train.jsonl"    # ← 修改为你的 jsonl 文件路径

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = "cuda:0"

split_type="train"
task_type="text_sql"
select_layer=24
save_dir = f"attention_hidden_data/Text-sql/Mistral-7B-Instruct-v0.3/{split_type}_layer{select_layer}_{task_type}"
os.makedirs(save_dir, exist_ok=True)

# Load your dataset
dataset = []
with open(data_path, "r") as f:
    for line in f:
        dataset.append(json.loads(line.strip()))

data_len = 0

for sample_i, sample in tqdm(enumerate(dataset), desc="Processing",total=len(dataset)):

    prompt = sample["prompt"]
    response = sample["response"]
    hallucination_label = sample["label"]   # ⭐ 这里直接拿一个 label

    # Build chat structure
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    input_text = text + response

    # tokenize
    input_ids = tokenizer([input_text], return_tensors="pt").input_ids.to(device)
    prefix_ids = tokenizer([text], return_tensors="pt").input_ids.to(device)
    print("Token length:",input_ids.shape)
    # Run model
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            return_dict=True,
            output_attentions=True,
            output_hidden_states=True
        )

    # hidden_states = outputs.hidden_states[select_layer]

    all_attentions = torch.stack(
        [att.squeeze(0).cpu() for att in outputs.attentions],
        dim=0
    )

    response_idx = prefix_ids.shape[-1]

    # decode tokens for inspection
    # V = []
    # for ids in input_ids.squeeze(0).cpu():
    #     V.append(tokenizer.decode([ids], skip_special_tokens=False))

    # ⭐ Final saved object — now only one hallucination_label
    save_obj = {
        "response_idx": response_idx,
        "token_ids": input_ids.squeeze(0).cpu(),  # (seq,)
        # "hidden_states": hidden_states.squeeze(0).cpu(),# (seq, dim)
        "attention": all_attentions,# (layers,heads, seq, seq)
        "hallucination_labels": hallucination_label,   #只保存一个 label
        "original_idx": sample_i,
        # "V": V,
    }
   
    save_path = f"{save_dir}/sample_{sample_i}.pt"
    torch.save(save_obj, save_path)
    torch.cuda.empty_cache()

    data_len += 1

print(f"保存{data_len}条数据到{save_dir}")
