import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from peft import LoraConfig, PeftModel
from utils import get_prompt, get_bnb_config
from tqdm import tqdm
import json
import os

def generate(args):
    bnb_config = get_bnb_config()
    lora_model = LoraConfig.from_pretrained(args.peft_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, quantization_config=bnb_config, device_map='auto')
    model = PeftModel(model, lora_model)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open(args.test_file, 'r') as f:
        examples = json.load(f)
    
    outputs = []
    for ex in tqdm(examples):
        result = {}
        prompt = get_prompt(ex['instruction'])
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids
        generation = model.generate(input_ids.to(model.device), max_new_tokens=args.max_length, num_return_sequences=1)
        result['id'] = ex['id']
        output = tokenizer.decode(generation[0], skip_special_tokens=True)[len(prompt):].strip()
        result['output'] = output
        outputs.append(result)
    
    with open(args.output_file, 'w') as f:
        json.dump(outputs, f, indent=4, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--peft_model', type=str, required=True)
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--max_length', type=int, default=128)
    args = parser.parse_args()
    generate(args)

if __name__ == '__main__':
    main()
