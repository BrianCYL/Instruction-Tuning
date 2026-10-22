from utils import get_fewshots
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import torch
import json
import argparse
from ppl import perplexity

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default="./data/public_test.json")
    parser.add_argument('--output_file', type=str, default="fewshot_output.json")
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained("zake7749/gemma-2-2b-it-chinese-kyara-dpo").to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("zake7749/gemma-2-2b-it-chinese-kyara-dpo")

    with open("./data/public_test.json", "r") as f:
        examples = json.load(f)
    
    ppl = perplexity(model, tokenizer, examples)
    print(ppl['mean_perplexity'])

    # with open(args.output_file, 'w') as f:
    #     json.dump(outputs, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main()