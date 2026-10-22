import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, get_scheduler
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
import argparse
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from utils import get_prompt, get_bnb_config
from ppl import perplexity
import math
from accelerate import Accelerator
from tqdm import tqdm
import numpy as np
import json
import os

class QLoRA_Trainer():
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.bnb_config = get_bnb_config()
        self.model = AutoModelForCausalLM.from_pretrained(self.args.model_path, quantization_config=self.bnb_config)
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_model(self):
        config = LoraConfig(
                            r=self.args.lora_r, 
                            lora_alpha=self.args.lora_alpha, # default=32 
                            target_modules=["q_proj", "v_proj"],
                            lora_dropout=self.args.lora_dropout,  # default=0.1
                            bias="none", 
                            task_type="CAUSAL_LM"
                        )
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, config)
        if self.args.verbose:
            self.model.print_trainable_parameters()
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.lr)

    def preprocess(self, examples):
        instructions = examples['instruction']
        outputs = examples['output']
        inputs = [
            self.tokenizer.bos_token+get_prompt(instruction)+output+self.tokenizer.eos_token for instruction, output in zip(instructions, outputs)
        ]
        model_inputs = self.tokenizer(inputs, padding="max_length", truncation=True, max_length=self.args.max_input_length, return_tensors="pt")
        labels = self.tokenizer(outputs, max_length=self.args.max_out_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def load_data(self):
        train_dataset = load_dataset("json", data_files=self.args.train_file_dir)
        train_dataset = train_dataset.map(self.preprocess, batched=True, remove_columns=train_dataset['train'].column_names)
        with open(self.args.test_file_dir, "r") as f:
            self.eval_dataset = json.load(f)
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        self.train_dataloader = DataLoader(
            train_dataset["train"],
            shuffle=True,
            collate_fn=self.data_collator,
            batch_size=self.args.train_batch_size,
        )
        
    
    def train(self):
        accelerator_log_kwargs = {}
        accelerator = Accelerator(gradient_accumulation_steps=self.args.gradient_accumulation_steps, **accelerator_log_kwargs)
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        max_train_steps = self.args.n_epochs * num_update_steps_per_epoch
        self.args.n_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=self.args.warmup_ratio * max_train_steps,
            num_training_steps=max_train_steps,
        )

        self.model.to(accelerator.device)
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.lr_scheduler
        )
        

        progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
        best_ppl = float('inf')
        ppl_curve = []
        for epoch in range(self.args.n_epochs):
            model.train()
            train_loss = 0
            optimizer.zero_grad()
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(model):
                    input_ids = batch["input_ids"]
                    attention_mask = batch["attention_mask"]
                    labels = batch["labels"]

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    loss = loss / self.args.gradient_accumulation_steps
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    train_loss += loss.item()
                if accelerator.sync_gradients:
                    progress_bar.update(1)
            # avg_train_loss = train_loss / len(train_dataloader)
            # accelerator.print(f"epoch {epoch}: {avg_train_loss}")
 
            model.eval()
            ppl = perplexity(model, self.tokenizer, self.eval_dataset)
            ppl_curve.append(ppl['mean_perplexity'])
            if ppl['mean_perplexity'] < best_ppl:
                best_ppl = ppl['mean_perplexity']
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(self.args.output_dir, save_function=accelerator.save)
                if accelerator.is_main_process:
                    self.tokenizer.save_pretrained(self.args.output_dir)  
            accelerator.print(f"epoch {epoch}: {ppl['mean_perplexity']}")
        accelerator.print(f"best ppl: {best_ppl}")
        json.dump(ppl_curve, open(os.path.join(self.args.output_dir, "ppl_tracking.json"), 'w'))
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="zake7749/gemma-2-2b-it-chinese-kyara-dpo")
    parser.add_argument("--train_file_dir", type=str, default="train.json")
    parser.add_argument("--test_file_dir", type=str, default="public_test.json")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_input_length", type=int, default=256)
    parser.add_argument("--max_out_length", type=int, default=256)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--verbose", default=True, action="store_true")
    args = parser.parse_args()
    trainer = QLoRA_Trainer(args)
    trainer.load_model()
    trainer.load_data()
    trainer.train()

if __name__ == "__main__":
    main()
    


